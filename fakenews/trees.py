
import re
import random
import jgrapht

def _lookup_RT(text):
    match = re.search(r'RT\s@((\w){1,15}):', text)
    if match: 
        return match.group(1)
    return None


def _find_retweet_source(dataset, retweet, previous_retweets):
    """Given a retweet and all previous retweers estimate from which 
    retweet it originated.
    """
    user = retweet.user
    rt_username = _lookup_RT(retweet.text)

    # Find a tweet from the RT user
    if rt_username is not None:
        candidates = []
        for rt in previous_retweets:
            if rt.user.screenname == rt_username: 
                candidates.append(rt)

        if len(candidates) != 0:
            return max(candidates, key= lambda k: rt.user.popularity)

    # Check if we follow some of the previous users that retweeted
    candidates = []
    for rt in previous_retweets:
        if user.id in rt.user.followers:
            candidates.append(rt)

    if len(candidates) != 0:
        return max(candidates, key= lambda k: rt.user.popularity)

    # Assign to most popular based on popularity
    weights = [rt.user.popularity for rt in previous_retweets]
    return random.choices(previous_retweets, weights=weights, k=1)[0]


def create_tree(dataset, tweet_id: str, min_retweets=5):
    """Given a tweet id, create a tree.
    """
    tweet = dataset.tweets_by_id[tweet_id]
    if tweet.is_retweet: 
        raise ValueError('Tree cannot be created from a retweet')


    if len(tweet.retweeted_by) < min_retweets:
        return None

    retweets = sorted(
        tweet.retweeted_by, key=lambda t: t.created_at, reverse=True
    )

    tree = jgrapht.create_graph(directed=True, any_hashable=True)
    tree.add_vertex(vertex=tweet)
    tree.vertex_attrs[tweet]['delay'] = 0

    previous = []
    previous.append(tweet)

    while len(retweets) != 0: 
        cur = retweets.pop()
        tree.add_vertex(vertex=cur)

        cur_retweet_of = _find_retweet_source(dataset, cur, previous)
        tree.add_edge(cur, cur_retweet_of)

        tree.vertex_attrs[cur]['delay'] = abs((cur.created_at-cur_retweet_of.created_at).total_seconds())

        previous.append(cur)

    if tweet.real: 
        tree.graph_attrs['label'] = "real"
    else:
        tree.graph_attrs['label'] = "fake"

    return tree


def postprocess_tree(tree):
    """Given a tree convert vertices to integers and compute features.
    """
    p_tree = jgrapht.create_graph(directed=True, any_hashable=True)

    vid = 0
    tweet_to_id = {}
    for tweet in tree.vertices:
        p_tree.add_vertex(vertex=vid)

        p_tree.vertex_attrs[vid]['user_id'] = tweet.user.id
        p_tree.vertex_attrs[vid]['delay'] = tree.vertex_attrs[tweet]['delay']
        p_tree.vertex_attrs[vid]['followers_count'] = max(len(tweet.user.followers), tweet.user.followers_count)
        p_tree.vertex_attrs[vid]['following_count'] =  max(len(tweet.user.following), tweet.user.following_count)

        for key in ['verified', 'protected', 'favourites_count', 'listed_count', 'statuses_count']:
            p_tree.vertex_attrs[vid][key] = int(getattr(tweet.user, key))

        #if tweet.user.embedding is not None:
        #    p_tree.vertex_attrs[vid]['user_profile_embedding'] = tweet.user.embedding

        tweet_to_id[tweet] = vid
        vid += 1
        
    for e in tree.edges:
        u = tree.edge_source(e)
        v = tree.edge_target(e)

        p_tree.add_edge(tweet_to_id[u], tweet_to_id[v])

    p_tree.graph_attrs['label'] = tree.graph_attrs['label']

    return p_tree



def create_trees(dataset, min_retweets=5):
    """Given a dataset create all tree
    """

    for tweet in dataset.tweets_by_id.values(): 
        if not tweet.is_retweet:
            tree = create_tree(dataset, tweet.id, min_retweets=min_retweets)
            if tree is not None:
                yield postprocess_tree(tree)


def tree_to_dict(tree):
    out = {}
    out['label'] = tree.graph_attrs['label']

    nodes = []
    for v in tree.vertices:
        node = {}
        node['id'] = v
        for k, v in tree.vertex_attrs[v].items():
            node[k] = v
        nodes.append(node)
    edges = []
    for e in tree.edges:
        edge = {}
        edge['source'] = tree.edge_source(e)
        edge['target'] = tree.edge_target(e)
        edges.append(edge)
    out['nodes'] = nodes
    out['edges'] = edges

    return out

def tree_to_json(tree):
    out = "{"
    out += "\"label\":\"{}\"".format(tree.graph_attrs['label'])
    out += ",\"nodes\":["
    first = True
    for v in tree.vertices:
        if first: 
            first = False
        else:
            out += ","
        out += "{"    
        out += "\"id\":\"{}\"".format(v)
        for k, v in tree.vertex_attrs[v].items():
            out += ",\"{}\":\"{}\"".format(k, v)
        out += "}"        
    out += "]"
    out += ",\"edges\":["
    first = True
    for e in tree.edges:
        if first: 
            first = False
        else:
            out += ","
        out += "{"
        out += "\"source\":\"{}\"".format(tree.edge_source(e))
        out += ",\"target\":\"{}\"".format(tree.edge_target(e))
        out += "}"
    out += "]"
    out += "}"
    return out
