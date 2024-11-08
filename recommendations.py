import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS
import warnings
import random
warnings.filterwarnings('ignore')

knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('music_data.csv')

non_feature_columns = ['track_id', 'artists', 'track_name']
features = df.select_dtypes(include=[np.number]).columns.difference(non_feature_columns)

fallback_recommendations = [
    {'track_name': 'Heat Waves', 'artists': 'Glass Animals'},
    {'track_name': 'Sunflower', 'artists': 'Post Malone, Swae Lee'},
    {'track_name': 'Electric Feel', 'artists': 'MGMT'},
    {'track_name': 'S.O.S.', 'artists': 'ABBA'},
    {'track_name': 'Under Control', 'artists': 'The Strokes'},
    {'track_name': 'Ain’t No Rest for the Wicked', 'artists': 'Cage the Elephant'},
    {'track_name': 'Take Me Out', 'artists': 'Franz Ferdinand'},
    {'track_name': 'Do I Wanna Know?', 'artists': 'Arctic Monkeys'},
    {'track_name': '1901', 'artists': 'Phoenix'},
    {'track_name': 'New Slang', 'artists': 'The Shins'},
    {'track_name': 'Taro', 'artists': 'Alt-J'},
    {'track_name': 'Young Folks', 'artists': 'Peter Bjorn and John'},
    {'track_name': 'We Are the People', 'artists': 'Empire of the Sun'},
    {'track_name': 'Fluorescent Adolescent', 'artists': 'Arctic Monkeys'},
    {'track_name': 'Lisztomania', 'artists': 'Phoenix'},
    {'track_name': 'The Less I Know the Better', 'artists': 'Tame Impala'},
    {'track_name': 'Another One Bites the Dust', 'artists': 'Queen'},
    {'track_name': 'Black Sheep', 'artists': 'Metric'},
    {'track_name': 'Go! (feat. Q-Tip)', 'artists': 'The Chemical Brothers'},
    {'track_name': 'Blue Sunday', 'artists': 'The Black Keys'},
    {'track_name': 'Safe and Sound', 'artists': 'Capital Cities'},
    {'track_name': 'The Wolf', 'artists': 'Bauhaus'},
    {'track_name': 'Linger', 'artists': 'The Cranberries'},
    {'track_name': 'Criminal', 'artists': 'Fiona Apple'},
    {'track_name': 'Young Lion', 'artists': 'Vampire Weekend'},
    {'track_name': 'Breezeblocks', 'artists': 'Alt-J'},
    {'track_name': 'Somebody Else', 'artists': 'The 1975'},
    {'track_name': 'Heartbeats', 'artists': 'The Knife'},
    {'track_name': 'Time Is on My Side', 'artists': 'The Rolling Stones'},
    {'track_name': 'The Night We Met', 'artists': 'Lord Huron'},
    {'track_name': 'Clint Eastwood', 'artists': 'Gorillaz'},
    {'track_name': 'Rude', 'artists': 'MAGIC!'},
    {'track_name': 'Feels Like We Only Go Backwards', 'artists': 'Tame Impala'},
    {'track_name': 'Don’t Delete the Kisses', 'artists': 'Wolf Alice'},
    {'track_name': 'Out of My League', 'artists': 'Fitz and The Tantrums'},
    {'track_name': 'Home', 'artists': 'Edward Sharpe & The Magnetic Zeros'},
    {'track_name': 'The Ocean', 'artists': 'Mike Perry'},
    {'track_name': 'Satellite', 'artists': 'Guster'},
    {'track_name': 'Electric Love', 'artists': 'BØRNS'},
    {'track_name': 'Shake It Out', 'artists': 'Florence + The Machine'},
    {'track_name': 'Little Dark Age', 'artists': 'MGMT'},
    {'track_name': 'Miracle Mile', 'artists': 'Cold War Kids'},
    {'track_name': 'Sweet Disposition', 'artists': 'The Temper Trap'},
    {'track_name': 'The Power of Love', 'artists': 'Huey Lewis & The News'},
    {'track_name': 'Tonight Tonight', 'artists': 'The Smashing Pumpkins'},
    {'track_name': 'Animal', 'artists': 'Neon Trees'},
    {'track_name': 'Future Starts Slow', 'artists': 'The Kills'},
    {'track_name': 'If I Had a Heart', 'artists': 'Fever Ray'},
    {'track_name': 'Shut Up and Dance', 'artists': 'Walk The Moon'},
    {'track_name': 'Take a Walk', 'artists': 'Passion Pit'},
    {'track_name': 'Cigarette Daydreams', 'artists': 'Cage the Elephant'},
    {'track_name': 'My Body', 'artists': 'Young the Giant'},
    {'track_name': 'Sedona', 'artists': 'Houndmouth'},
    {'track_name': 'Pompeii', 'artists': 'Bastille'},
    {'track_name': 'The Joke', 'artists': 'Brandi Carlile'},
    {'track_name': 'We Don’t Talk Anymore', 'artists': 'Charlie Puth ft. Selena Gomez'},
    {'track_name': 'Boys Don’t Cry', 'artists': 'The Cure'},
    {'track_name': 'Somewhere Only We Know', 'artists': 'Keane'},
    {'track_name': 'Kiss Me', 'artists': 'Sixpence None the Richer'},
    {'track_name': 'Rivers and Roads', 'artists': 'The Head and the Heart'},
    {'track_name': 'Dog Days Are Over', 'artists': 'Florence + The Machine'},
    {'track_name': 'Junk of the Heart (Happy)', 'artists': 'The Kooks'},
    {'track_name': 'Wide Eyes', 'artists': 'The Local Natives'},
    {'track_name': 'Cecilia and the Satellite', 'artists': 'Andrew McMahon in the Wilderness'},
    {'track_name': 'Sigh No More', 'artists': 'Mumford & Sons'},
    {'track_name': 'The Sound of Silence', 'artists': 'Simon & Garfunkel'},
    {'track_name': 'Chasing Cars', 'artists': 'Snow Patrol'},
    {'track_name': 'Fell In Love With A Girl', 'artists': 'The White Stripes'},
    {'track_name': 'The Middle', 'artists': 'Jimmy Eat World'},
    {'track_name': 'Pumped Up Kicks', 'artists': 'Foster The People'},
    {'track_name': 'Strange Mercy', 'artists': 'St. Vincent'},
    {'track_name': 'Call It What You Want', 'artists': 'Taylor Swift'},
    {'track_name': 'One Last Time', 'artists': 'Ariana Grande'},
    {'track_name': 'Tear in My Heart', 'artists': 'Twenty One Pilots'},
    {'track_name': 'A-Punk', 'artists': 'Vampire Weekend'},
    {'track_name': 'Take It Easy', 'artists': 'The Eagles'},
    {'track_name': 'The Suburbs', 'artists': 'Arcade Fire'},
    {'track_name': 'I Wanna Be Your Lover', 'artists': 'Prince'},
    {'track_name': 'First', 'artists': 'Cold War Kids'},
    {'track_name': 'Some Nights', 'artists': 'fun.'},
    {'track_name': '1901', 'artists': 'Phoenix'},
    {'track_name': 'The Wire', 'artists': 'Haim'}
]


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def get_recommendations(track_name, df, knn_model, n=10):
    try:
        idx = df.index[df['track_name'].str.lower() == track_name.lower()].tolist()[0]
    except IndexError:
        track_name_parts = track_name.split()
        while track_name_parts and len(track_name_parts) > 0:
            track_name_parts.pop()  
            new_track_name = " ".join(track_name_parts)
            try:
                idx = df.index[df['track_name'].str.lower() == new_track_name.lower()].tolist()[0]
                break  
            except IndexError:
                continue  
        else:
            return random.sample(fallback_recommendations, n)
    
    song_features = df[features].iloc[idx].values.reshape(1, -1)
    song_features_scaled = scaler.transform(song_features)

    distances, indices = knn_model.kneighbors(song_features_scaled)

    unique_recommendations = set()
    recommendations = []

    for index in indices.flatten()[1:]:  
        if len(recommendations) < n and df.iloc[index]['track_name'] not in unique_recommendations:
            unique_recommendations.add(df.iloc[index]['track_name'])
            recommendations.append(df.iloc[index][['track_name', 'artists']].to_dict())

    return recommendations

@app.route('/get_recommendations', methods=['OPTIONS', 'GET'])
def recommend():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'CORS preflight check passed'}), 200
    song_name = request.args.get('song_name')
    if not song_name:
        return jsonify({'error': 'Song name is required'}), 400

    recommendations = get_recommendations(song_name.lower(), df, knn_model)

    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
