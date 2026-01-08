import gradio as gr
import ray
import time
import yaml
import asyncio
from typing import List, Dict, Tuple, Optional
import ollama
from enum import Enum


class ResearchChatbot:
    def __init__(self, config_path: str = "config/cluster_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        master_host = self.config['master_node']['host']
        master_port = self.config['master_node']['ray_port']
        redis_password = self.config['master_node']['redis_password']
        
        ray.init(
            address=f"ray://{master_host}:{master_port}",
            _redis_password=redis_password
        )
        
        print(f"Connected to Ray cluster at {master_host}:{master_port}")
        
        # LLM config
        llm_config = self.config['llm_frontend']['llm']
        self.llm_model = llm_config['model']
        self.temperature = llm_config['temperature']
        self.max_tokens = llm_config['max_tokens']
        self.system_prompt = llm_config['system_prompt']
        
        # Initialize Ollama
        try:
            ollama.list()  # Test connection
            print(f"Ollama connected, using model: {self.llm_model}")
        except Exception as e:
            print(f"Ollama not available: {e}")
        
        # Get master node actors
        try:
            self.scheduler = ray.get_actor("scheduler")
            self.metrics = ray.get_actor("metrics")
            print("✓ Connected to Ray actors")
        except Exception as e:
            print(f"⚠ Could not connect to Ray actors: {e}")
            self.scheduler = None
            self.metrics = None
        
        print("✓ ResearchChatbot initialized")
    
    def parse_user_intent(self, message: str) -> Dict:
        """
        Use LLM to understand user intent.
        
        Returns:
            Dict with parsed intent (query_type, keywords, operation)
        """
        prompt = f"""Analyze this research query and extract structured information:

                User Query: "{message}"

                Determine:
                1. query_type: "paper_search" | "author_search" | "topic_exploration" | "citation_analysis"
                2. keywords: List of main research topics/keywords
                3. operation: "find_papers" | "find_citations" | "find_references" | "find_related" | "train_model"
                4. entities: Any specific paper titles or author names mentioned
                5. filters: Year ranges, venues, etc.

                Respond in valid JSON format only, no explanation."""
        
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.3, 
                    'num_predict': 200
                }
            )
        
            import json
            intent = json.loads(response['response'])
            return intent
        
        except Exception as e:
            print(f"Intent parsing failed: {e}")
            return {
                'query_type': 'topic_exploration',
                'keywords': [message],
                'operation': 'find_papers',
                'entities': [],
                'filters': {}
            }
    
    def format_results_naturally(
        self,
        user_query: str,
        results: Dict,
        job_type: str
    ) -> str:
        """
        Use LLM to format results in natural language.
        
        Args:
            user_query: Original user query
            results: Job results from Ray worker
            job_type: Type of job ("rl_training" or "rag_inference")
        
        Returns:
            Natural language response
        """
        if job_type == "rl_training":
            prompt = f"""The user asked: "{user_query}"

            We trained an RL agent and found relevant research papers. Here are the results:

            Training Results:
            - Episodes trained: {results.get('episodes', 0)}
            - Average reward: {results.get('avg_reward', 0):.2f}
            - Max similarity achieved: {results.get('max_similarity', 0):.3f}
            - Training time: {results.get('duration_sec', 0):.1f} seconds

            Generate a friendly, conversational response (2-3 sentences) explaining:
            1. What we found
            2. How relevant the results are (based on similarity score)
            3. Suggest next steps

            Be natural and helpful, like talking to a colleague."""
                
        else: 
            prompt = f"""The user asked: "{user_query}"

        We found {results.get('num_papers', 0)} relevant papers through graph traversal:

        Results:
        - Papers explored: {results.get('num_papers', 0)}
        - Average relevance: {results.get('avg_similarity', 0):.3f}
        - Inference time: {results.get('duration_sec', 0):.2f} seconds

        Generate a friendly response explaining what we found and how relevant it is."""
        
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': 150
                }
            )
            
            return response['response']
        
        except Exception as e:
            print(f"LLM formatting failed: {e}")
            # Fallback to template
            return self._fallback_format(results, job_type)
    
    def _fallback_format(self, results: Dict, job_type: str) -> str:
        """Fallback formatting without LLM."""
        if job_type == "rl_training":
            return f"""Training Complete!

            I trained an RL agent and explored the citation graph. Here's what I found:

            **Training Stats:**
            - Trained for {results.get('episodes', 0)} episodes
            - Achieved {results.get('max_similarity', 0):.1%} similarity to your query
            - Average reward: {results.get('avg_reward', 0):.2f}

            The agent explored {results.get('episodes', 0)} different research paths and found papers with {results.get('avg_similarity', 0):.1%} average relevance.

            Would you like me to show you the top recommended papers?"""
        
        else:
            return f"""Found {results.get('num_papers', 0)} papers!

            Average relevance: {results.get('avg_similarity', 0):.1%}
            Search time: {results.get('duration_sec', 0):.1f}s"""
    
    def chat(
        self,
        message: str,
        history: List[Tuple[str, str]]
    ) -> Tuple[str, List[Tuple[str, str]]]:
        if not self.scheduler:
            history.append((message, "Ray scheduler not available. Please check cluster connection."))
            return "", history
        
        # Parse user intent
        intent = self.parse_user_intent(message)
        
        # Determine if we should train or just infer
        operation = intent.get('operation', 'find_papers')
        
        if operation == 'train_model' or 'train' in message.lower():
            job_type = 'rl_training'
            response_prefix = "Starting RL training... This will take a few minutes.\n\n"
        else:
            job_type = 'rag_inference'
            response_prefix = "Searching for papers...\n\n"
        
        # Create initial response
        history.append((message, response_prefix + "Processing..."))
        yield "", history
        
        # Submit job to Ray cluster
        try:
            from core.job_spec import create_rl_training_job
            
            keywords = ' '.join(intent.get('keywords', [message]))
            
            if job_type == 'rl_training':
                job = create_rl_training_job(
                    episodes=100,
                    query=keywords,
                    start_paper_id="arxiv_1706.03762"  
                )
            
            # Submit to scheduler
            job_id = ray.get(self.scheduler.submit_job.remote(job))
            
            max_wait = 300  
            start_time = time.time()
            
            while (time.time() - start_time) < max_wait:
                status_result = ray.get(self.scheduler.get_job_status.remote(job_id))
                
                if status_result is not None:

                    status_value = status_result.value if hasattr(status_result, 'value') else str(status_result)
                    
                    if status_value == 'completed':
                        completed_jobs = ray.get(self.scheduler.get_completed_jobs.remote())
                        
                        job_data = completed_jobs.get(job_id)
                        if job_data and isinstance(job_data, dict):
                            results = job_data.get('result', {})
                        else:
                            results = {}
                    
                        natural_response = self.format_results_naturally(
                            message,
                            results,
                            job_type
                        )
                        
                        history[-1] = (message, response_prefix + natural_response)
                        yield "", history
                        break
                    
                    elif status_value == 'failed':
                        error_msg = " Job failed. Please try again or rephrase your query."
                        history[-1] = (message, error_msg)
                        yield "", history
                        break
                
                # ✅ FIXED: Proper time.sleep call
                time.sleep(2)
            
            else:
                # Timeout
                timeout_msg = "Request timed out. The job is still running in the background."
                history[-1] = (message, timeout_msg)
                yield "", history
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}\n\nPlease try again."
            history[-1] = (message, error_msg)
            yield "", history
        
        return "", history
    
    def launch(self):
        """Launch Gradio interface."""
        server_config = self.config['llm_frontend']['server']
        
        with gr.Blocks(
            title="Research Paper Recommendation Chat",
            theme=gr.themes.Soft()
        ) as demo:
            gr.Markdown("""
            # 📚 Research Paper Recommendation Assistant
            
            Ask me to find papers, explore topics, or train recommendation models!
            
            **Example queries:**
            - "Find papers about attention mechanisms in transformers"
            - "Show me recent work on reinforcement learning"
            - "Train a model to recommend papers about computer vision"
            - "What papers cite the Attention is All You Need paper?"
            """)
            
            chatbot = gr.Chatbot(
                label="Research Assistant",
                height=500,
                show_label=True,
                avatar_images=(None, "")
            )
            
            msg = gr.Textbox(
                label="Your Query",
                placeholder="Ask me about research papers...",
                lines=2
            )
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
            
            # Examples
            gr.Examples(
                examples=[
                    "Find papers about attention mechanisms",
                    "Train a model for NLP paper recommendations",
                    "What papers are related to BERT?",
                    "Show me recent work on graph neural networks"
                ],
                inputs=msg
            )
            
            # Event handlers
            msg.submit(
                self.chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            submit.click(
                self.chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear.click(
                lambda: (None, []),
                outputs=[msg, chatbot]
            )
        
        # Launch server
        demo.launch(
            server_name="0.0.0.0",
            server_port=server_config['port'],
            share=server_config.get('share', False),
            auth=tuple(server_config['auth']) if server_config.get('auth') else None
        )
        
        print(f"\n✓ Chat UI launched at http://{self.config['llm_frontend']['host']}:{server_config['port']}")


if __name__ == "__main__":
    chatbot = ResearchChatbot()
    chatbot.launch()
