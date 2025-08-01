import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Define the food categories we'll use
FOOD_CATEGORIES = [
    'apple pie', 'burger', 'cheesecake', 'chicken curry', 
    'chicken wings', 'chocolate cake', 'donuts', 'fries', 
    'hot dog', 'ice cream', 'pizza', 'sandwich', 
    'steak', 'tacos', 'waffles'
]

# Mapping between YOLO detection classes and our categories
YOLO_TO_CATEGORY = {
    'apple pie': 'apple pie',
    'burger': 'burger',
    'cheeseburger': 'burger',
    'pizza': 'pizza',
    'sandwich': 'sandwich',
    'hot dog': 'hot dog',
    'taco': 'tacos',
    'steak': 'steak',
    'fries': 'fries',
    'waffle': 'waffles',
    'ice cream': 'ice cream',
    'donut': 'donuts',
    'cheesecake': 'cheesecake',
    'chicken wings': 'chicken wings',
    'chicken curry': 'chicken curry',
    'chocolate cake': 'chocolate cake'
}

def map_to_categories(detected_items):
    """Map YOLO detected items to our food categories"""
    categories = []
    for item in detected_items:
        # Try exact match first
        if item in YOLO_TO_CATEGORY:
            categories.append(YOLO_TO_CATEGORY[item])
        else:
            # Try partial matching
            for yolo_class, category in YOLO_TO_CATEGORY.items():
                if yolo_class in item:
                    categories.append(category)
                    break
    return categories

class RestaurantRecommender:
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's folder
    data_path = os.path.join(script_dir, "irish_restaurants_meals_final.csv")  # Construct full path
    def __init__(self, data_path = data_path):
        # Load and preprocess data
        self.df = pd.read_csv(data_path)
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Clean and prepare the restaurant data"""
        # Data cleaning
        self.df.dropna(subset=['Rating', 'Price', 'Latitude', 'Longitude'], inplace=True)
        
        # Add category column
        self.df['Category'] = self.df['Meal'].apply(self._categorize_meal)
        
        # Feature scaling
        scaler = StandardScaler()
        features = self.df[['Price', 'Rating', 'Latitude', 'Longitude']]
        self.df[['Price_scaled', 'Rating_scaled', 'Lat_scaled', 'Lon_scaled']] = scaler.fit_transform(features)
        
        # Precompute similarity matrix
        self.feature_matrix = self.df[['Price_scaled', 'Rating_scaled', 'Lat_scaled', 'Lon_scaled']].values
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
    
    def _categorize_meal(self, meal_name):
        """Categorize a meal into one of our predefined categories"""
        meal_lower = meal_name.lower()
        for category in FOOD_CATEGORIES:
            if category in meal_lower:
                return category
        return 'other'
    
    def recommend_by_category(self, categories, top_n=5):
        if not categories:
        # Return top rated unique restaurants if no categories detected
                 return self.df.drop_duplicates('Restaurant').nlargest(top_n, 'Rating')
    
        # Get restaurants that match ANY of the detected categories
        relevant = self.df[self.df['Category'].isin(categories)]
    
        if len(relevant) == 0:
            return self.df.drop_duplicates('Restaurant').nlargest(top_n, 'Rating')
    
        # Score restaurants by how many categories they match
        category_counts = Counter(categories)
    
        def score_restaurant(row):
            score = 0
            for category, count in category_counts.items():
                if row['Category'] == category:
                    score += count
            return score
    
        relevant['MatchScore'] = relevant.apply(score_restaurant, axis=1)
    
    # Get unique restaurants sorted by best match and rating
        unique_restaurants = (relevant
                            .sort_values(['MatchScore', 'Rating'], ascending=[False, False])
                            .drop_duplicates('Restaurant')
                            .head(top_n))
    
        return unique_restaurants
    def recommend_by_restaurant(self, restaurant_name, top_n=5):
        """Recommend similar restaurants to a given one"""
        if restaurant_name not in self.df['Restaurant'].values:
            return self.df.nlargest(top_n, 'Rating')
            
        idx = self.df[self.df['Restaurant'] == restaurant_name].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]  # Exclude self and get top_n
        
        restaurant_indices = [i[0] for i in sim_scores]
        return self.df.iloc[restaurant_indices]
    
    def get_recommendations(self, detected_items=None, restaurant_name=None, top_n=5):
        """Main method to get recommendations"""
        if detected_items:
            categories = map_to_categories(detected_items)
            print(f"Mapped detected items to categories: {categories}")
            return self.recommend_by_category(categories, top_n)
        elif restaurant_name:
            return self.recommend_by_restaurant(restaurant_name, top_n)
        else:
            return self.df.nlargest(top_n, 'Rating')

if __name__ == "__main__":
    # For testing the recommender standalone
    recommender = RestaurantRecommender()
    
    # Test category-based recommendation
    print("\nTesting category-based recommendation for ['burger', 'fries']:")
    print(recommender.get_recommendations(detected_items=['burger', 'fries']).head())
    
    # Test restaurant-based recommendation
    print("\nTesting restaurant-based recommendation for 'Burger King':")
    print(recommender.get_recommendations(restaurant_name="Burger King").head())
    
    # Test default recommendation
    print("\nTesting default recommendation:")
    print(recommender.get_recommendations().head())