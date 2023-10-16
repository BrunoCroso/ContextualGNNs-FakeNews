
class Tweet: 

    def __init__(self, id): 
        self._id = id
        self._user = None
        self._created_at = None
        self._text = None
        self._retweet_of = None
        self._retweeted_by = []
        self._real = None

    @property
    def id(self):
        return self._id

    @property
    def created_at(self):
        return self._created_at

    @created_at.setter
    def created_at(self, value):
        self._created_at = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def real(self):
        return self._real

    @real.setter
    def real(self, value):
        self._real = value

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, value):
        self._user = value

    @property
    def is_retweet(self):
        return self._retweet_of is not None

    @property
    def retweet_of(self):
        return self._retweet_of

    @retweet_of.setter
    def retweet_of(self, value):
        self._retweet_of = value

    @property
    def retweeted_by(self):
        return self._retweeted_by

    def __repr__(self):
        return "Tweet(%r)" % self._id


class User: 

    def __init__(self, id):
        self._id = id
        self._followers = set()
        self._following = set()
        self._followers_count = 0
        self._following_count = 0
        self._listed_count = 0
        self._favourites_count = 0
        self._statuses_count = 0
        self._verified = False
        self._protected = False
        self._screenname = None
        self._description = None
        self._embedding = None

    @property
    def id(self):
        return self._id
    
    @property
    def followers(self):
        return self._followers

    @property
    def following(self) :
        return self._following

    @property
    def screenname(self):
        return self._screenname

    @screenname.setter
    def screenname(self, value):
        self._screenname = value

    @property
    def popularity(self):
        return len(self._followers)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def verified(self):
        return self._verified

    @verified.setter
    def verified(self, value) :
        self._verified = value

    @property
    def protected(self):
        return self._protected

    @protected.setter
    def protected(self, value) :
        self._protected = value

    @property
    def followers_count(self):
        return self._followers_count

    @followers_count.setter
    def followers_count(self, value) :
        self._followers_count = value

    @property
    def following_count(self):
        return self._following_count

    @following_count.setter
    def following_count(self, value) :
        self._following_count = value

    @property
    def statuses_count(self):
        return self._statuses_count

    @statuses_count.setter
    def statuses_count(self, value) :
        self._statuses_count = value

    @property
    def listed_count(self):
        return self._listed_count

    @listed_count.setter
    def listed_count(self, value) :
        self._listed_count = value

    @property
    def favourites_count(self):
        return self._favourites_count

    @favourites_count.setter
    def favourites_count(self, value) :
        self._favourites_count = value

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, value) :
        self._embedding = value

    def __repr__(self):
        return "User(%r)" % self._id

    
