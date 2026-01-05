#import "@preview/modern-cv:0.9.0": *

#show: resume.with(
  author: (
    firstname: "Kausik",
    lastname: "Muthukumar",
    email: "kaushik.80405@gmail.com",
    github: "SycamoreQ",
    birth: "April 8th, 2005",
    linkedin: "kaushikmuthu",
    phone: "(+91) 9880838334", 
    positions: ("Student",),
  ),
  profile-picture: none,
  // date: datetime.today().display(),
  language: "en",
  colored-headers: true,
  show-footer: true,
  paper-size: "a4",
)

#show link: underline

= About Me
Kausik Muthukumar is a third-year AI and Data Science student with technical expertise in Deep Learning, particularly in Graph Learning, Reinforcement Learning, and Quantum Computing.  
He aims to develop unique AI solutions and products using graph-based learning approaches and is seeking opportunities to apply his skills in building impactful technological systems.

---

= Education
#resume-entry(
  title: "Amrita School of Engineering",
  location: "Bangalore",
  date: "August 2023 – May 2027",
  description: "B.Tech in Artificial Intelligence & Data Science",
)
- CGPA: 7.9

---

= Projects

#resume-entry(
  title: "Epidemic Simulator",
  date: "July 2025 – Present",
  location: link("https://github.com/SycamoreQ/EpidemicSimulator")[github.com/SycamoreQ/EpidemicSimulator],
  description: "Developer"
)
- Implemented a real-time, scalable epidemic prediction software using the Double Deep Q-Network (DDQN) Reinforcement Learning algorithm in Scala.  
- Utilized Apache Spark for big data input, Spark Streaming, and Weights & Biases (WandB) for real-time visualization.

#resume-entry(
  title: "rust-graph",
  date: "September 2025 – Present",
  location: link("https://github.com/SycamoreQ/rust-graph")[github.com/SycamoreQ/rust-graph],
  description: "Developer"
)
- Implemented a graph library in Rust with support for Graph Convolutional Networks (GCN), Graph Attention Transformers (GAT), and Dynamic Graphs.
- Utilizes candle.rs , petgraph and ndarray libraries for deep learning and graph operations.

#resume-entry(
  title: "ResearchBot",
  date: "December 2025",
  location: link("https://github.com/SycamoreQ/5glab")[github.com/SycamoreQ/5glab],
  description: "Developer"
)
- Developed a RAG model trained on the DDQN algorithm to retrive research papers from a Knowledge Graph database.  
- Trained the model over a distributed cluster of GPUs using Ray.io. 

---

= Other Projects
- *#link("https://github.com/SycamoreQ/HyperGraphTransformer")[Vison Model]* – Hypergraph Transformer model for image classification using PyTorch.  
- *#link("https://github.com/nairadithya/pala")[Cloud Native Web App]* – Contributed to a cloud-native app gated behind a load balancer and reverse proxy.  
- *#link("https://github.com/SycamoreQ/queryscheduler")[Query Scheduler]* – Custom Query Scheduler based on Query Retrieval Depth written in Rust.

---

= Skills
#resume-skill-item(
  "Languages",
  (strong("Python"), strong("Rust"), strong("Scala"), "GoLang", "Java" , "Cypher"),
)

#resume-skill-item(
  "Libraries / Frameworks",
  (strong("PyTorch"), "candle.rs (Rust DL Library)", "Gin (Web Framework)", "Scalapy" , "Ray.io" , "Apache Spark" , "Neo4j"),
)


#resume-skill-item(
  "Software / Tools",
  (strong("Docker"), strong("Kubernetes"), strong("CUDA"), "Git", "LaTeX", "Typst"),
)


#resume-skill-item(
  "Core Areas",
  ("Artificial Intelligence", "Deep Learning", "Graph Neural Networks", "Reinforcement Learning", "Quantum Computing"),
)