import logging
import requests
from config import APIConfig
from typing import Any, Dict, List

logger = logging.getLogger("stock_predictor.sentiment")


class RealTimeSentimentAnalyzer:
    """Real-time sentiment analysis with actual API integrations"""

    def __init__(self):
        self.api_keys = APIConfig.get_all_keys()

    def _get_real_news_headlines(self, symbol: str) -> List[Dict]:
        """Get real news headlines from APIs"""
        headlines = []

        # Try NewsAPI
        if self.api_keys['newsapi']:
            try:
                url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.api_keys['newsapi']}&pageSize=10"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    headlines.extend([{
                        'title': article['title'],
                        'description': article['description'],
                        'source': article['source']['name'],
                        'published_at': article['publishedAt'],
                        'url': article['url']
                    } for article in articles])
            except Exception as e:
                logger.warning(f"NewsAPI failed: {e}")
        
        # Try Alpha Vantage
        if self.api_keys['alphavantage'] and not headlines:
            try:
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.api_keys['alphavantage']}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('feed', [])
                    headlines.extend([{
                        'title': article['title'],
                        'summary': article.get('summary', ''),
                        'source': article.get('source', 'Unknown'),
                        'time_published': article.get('time_published', ''),
                        'url': article.get('url', ''),
                        'sentiment_score': article.get('overall_sentiment_score', 0)
                    } for article in articles])
            except Exception as e:
                logger.warning(f"Alpha Vantage news failed: {e}")
        
        return headlines
    
    def _get_twitter_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Twitter sentiment for symbol"""
        if not self.api_keys['twitter']:
            return {'available': False, 'error': 'No Twitter API key'}
        
        try:
            # Twitter API v2 implementation
            headers = {
                'Authorization': f'Bearer {self.api_keys["twitter"]}'
            }
            
            # Search for recent tweets about the stock
            query = f"${symbol} OR {symbol} stock -is:retweet"
            url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=50"
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                tweets = data.get('data', [])
                
                # Analyze tweet sentiment
                sentiments = []
                for tweet in tweets:
                    sentiment = self._analyze_text_sentiment(tweet['text'])
                    sentiments.append(sentiment)
                
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5
                
                return {
                    'available': True,
                    'score': avg_sentiment,
                    'tweet_count': len(tweets),
                    'sentiment': self._get_sentiment_label(avg_sentiment)
                }
            
        except Exception as e:
            logger.warning(f"Twitter API failed: {e}")
        
        return {'available': False}
    
    def _get_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Reddit sentiment from investing subreddits"""
        if not self.api_keys['reddit']['client_id']:
            return {'available': False, 'error': 'No Reddit API keys'}
        
        try:
            # Reddit API implementation
            auth = requests.auth.HTTPBasicAuth(
                self.api_keys['reddit']['client_id'],
                self.api_keys['reddit']['client_secret']
            )
            
            data = {
                'grant_type': 'password',
                'username': self.api_keys['reddit'].get('username', ''),
                'password': self.api_keys['reddit'].get('password', '')
            }
            
            headers = {'User-Agent': 'StockPredictor/1.0'}
            
            # Get access token
            response = requests.post(
                'https://www.reddit.com/api/v1/access_token',
                auth=auth, data=data, headers=headers
            )
            
            if response.status_code == 200:
                token = response.json()['access_token']
                headers = {**headers, 'Authorization': f'bearer {token}'}
                
                # Search in investing subreddits
                subreddits = ['stocks', 'investing', 'wallstreetbets']
                all_posts = []
                
                for subreddit in subreddits:
                    url = f'https://oauth.reddit.com/r/{subreddit}/search?q={symbol}&restrict_sr=1'
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        posts = response.json().get('data', {}).get('children', [])
                        all_posts.extend(posts)
                
                # Analyze post sentiments
                sentiments = []
                for post in all_posts[:20]:  # Limit to 20 posts
                    title = post['data']['title']
                    sentiment = self._analyze_text_sentiment(title)
                    sentiments.append(sentiment)
                
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5
                
                return {
                    'available': True,
                    'score': avg_sentiment,
                    'post_count': len(all_posts),
                    'sentiment': self._get_sentiment_label(avg_sentiment)
                }
        
        except Exception as e:
            logger.warning(f"Reddit API failed: {e}")
        
        return {'available': False}