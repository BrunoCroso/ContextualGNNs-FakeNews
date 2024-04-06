import json

def main():

    # Read model configuration from options.json
    with open("options.json", "r") as json_file:
        options = json.load(json_file)

    # Extract information from the configuration

    if options["embedder_type"].lower() == "glove":
        embedder = "GloVe"  
    else:
        embedder = "BERTweet"

    # Display the model information
    print("Model Information:")
    print(f"Embedder used: {embedder}")
    print(f"Contains user embeddings: {options['user_embeddings']}")
    print(f"Contains retweet embeddings: {options['retweet_embeddings']}")
    print(f"Initial NEFTune noise alpha used: {options['initial_neftune_noise_alpha']}")
    print(f"NEFTune noise alpha step size used: {options['neftune_noise_alpha_step_size']}")
    print(f"Number of models that will be trained: {options['number_of_models']}\n")


if __name__ == "__main__":
    main()
