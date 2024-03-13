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
    print(f"NEFTune noise alpha used: {options['neftune_noise_alpha']}\n")


if __name__ == "__main__":
    main()
