from .ray_dist import run_training

if __name__ == "__main__":
    NUM_EXPLORERS = 8     
    TOTAL_EPISODES = 2000
    
    run_training(
        num_explorers=NUM_EXPLORERS, 
        total_eps=TOTAL_EPISODES
    )
