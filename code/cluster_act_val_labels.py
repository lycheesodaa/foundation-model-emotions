import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import re
from collections import Counter

def parse_emotion_file(content):
    """Parse emotion ratings from file content."""
    # Previous implementation remains the same
    utterances = {}
    for line in content.split('\n'):
        if not line.strip():
            continue
        match = re.match(r'(Ses\d+[MF]_\w+_[MF]\d+)\s+:act\s+(\d+);\s+:val\s+(\d+);', line)
        if match:
            utterance_id, act, val = match.groups()
            utterances[utterance_id] = {
                'activation': float(act),
                'valence': float(val)
            }
    return utterances

def compute_average_ratings():
    """Compute average ratings across evaluators."""
    # Previous implementation remains the same
    all_ratings = {}
    
    for session in range(1, 6):
        folderpath = f'Processed/IEMOCAP_full_release/Session{session}/dialog/EmoEvaluation/Attribute/'
        evaluator_files = [folderpath + filename for filename in os.listdir(folderpath) if filename]
        
        for file in evaluator_files:
            try:
                with open(file, 'r') as file:
                    content = file.read()
            except FileNotFoundError:
                print(f"Warning: Could not find file {file}")
            
            ratings = parse_emotion_file(content)
            for utterance_id, values in ratings.items():
                if utterance_id not in all_ratings:
                    all_ratings[utterance_id] = {
                        'activation': [],
                        'valence': []
                    }
                all_ratings[utterance_id]['activation'].append(values['activation'])
                all_ratings[utterance_id]['valence'].append(values['valence'])
    
    averaged_ratings = {}
    for utterance_id, values in all_ratings.items():
        averaged_ratings[utterance_id] = {
            'activation': np.mean(values['activation']),
            'valence': np.mean(values['valence'])
        }
    
    return averaged_ratings

def map_utterance_to_emotion(path):
    """Extract utterance ID and map to corresponding emotion from csv."""
    match = re.search(r'(Ses\d+[MF]_\w+_\d+_[MF]\d+)\.wav', path)
    return match.group(1) if match else None

def analyze_cluster_emotions(cluster_stats, emotion_df):
    """Analyze emotional content of each cluster."""
    # Create emotion category mappings
    emotion_categories = {
        'happiness/excitement': ['hap', 'exc'],
        'sadness/frustration': ['sad', 'fru'],
        'anger/frustration': ['ang', 'fru'],
        'neutrality': ['neu']
    }
    
    cluster_emotions = {}
    
    for cluster, stats in cluster_stats.items():
        # Get utterance IDs in this cluster
        utterances = stats['utterances']
        
        # Create mapping from utterance ID to emotion label
        utterance_emotions = {}
        for _, row in emotion_df.iterrows():
            utterance_id = map_utterance_to_emotion(row['path'])
            if utterance_id and row['agreement'] >= 2:  # Only consider labels with agreement >= 2
                utterance_emotions[utterance_id] = row['emotion']
        
        # Count emotions in this cluster
        emotion_counts = Counter()
        for utterance in utterances:
            if utterance in utterance_emotions:
                emotion = utterance_emotions[utterance]
                emotion_counts[emotion] += 1
        
        # Calculate category scores
        category_scores = {}
        total_labeled = sum(emotion_counts.values())
        
        for category, emotions in emotion_categories.items():
            score = sum(emotion_counts[e] for e in emotions) / total_labeled if total_labeled > 0 else 0
            category_scores[category] = score
        
        # Assign dominant category
        dominant_category = max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else "Unknown"
        
        cluster_emotions[cluster] = {
            'dominant_category': dominant_category,
            'category_scores': category_scores,
            'emotion_counts': dict(emotion_counts),
            'total_labeled': total_labeled,
            'avg_activation': stats['avg_activation'],
            'avg_valence': stats['avg_valence']
        }
    
    return cluster_emotions

def analyze_clusters(clustered_utterances, cluster_centers):
    """Analyze the characteristics of each cluster."""
    cluster_stats = {}
    for i in range(len(cluster_centers)):
        cluster_stats[i] = {
            'count': 0,
            'avg_activation': 0,
            'avg_valence': 0,
            'utterances': []
        }
    
    # Collect statistics
    for utterance_id, data in clustered_utterances.items():
        cluster = data['cluster']
        cluster_stats[cluster]['count'] += 1
        cluster_stats[cluster]['avg_activation'] += data['coordinates']['activation']
        cluster_stats[cluster]['avg_valence'] += data['coordinates']['valence']
        cluster_stats[cluster]['utterances'].append(utterance_id)
    
    # Compute averages
    for cluster in cluster_stats:
        count = cluster_stats[cluster]['count']
        if count > 0:
            cluster_stats[cluster]['avg_activation'] /= count
            cluster_stats[cluster]['avg_valence'] /= count
    
    return cluster_stats

def perform_clustering_with_labels(ratings, emotion_df, n_clusters=4):
    """Perform clustering and analyze emotion labels."""
    # Perform clustering
    points = np.array([[v['activation'], v['valence']] for v in ratings.values()])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(points)
    
    # Assign clusters to utterances
    clustered_utterances = {}
    for (utterance_id, _), cluster_id in zip(ratings.items(), clusters):
        clustered_utterances[utterance_id] = {
            'cluster': cluster_id,
            'coordinates': ratings[utterance_id]
        }
    
    # Analyze clusters
    cluster_stats = analyze_clusters(clustered_utterances, kmeans.cluster_centers_)
    
    # Analyze emotional content
    cluster_emotions = analyze_cluster_emotions(cluster_stats, emotion_df)
    
    return clustered_utterances, kmeans.cluster_centers_, cluster_emotions

# Example usage
if __name__ == "__main__":
    # Load emotion labels from CSV
    emotion_df = pd.read_csv('Processed\iemocap_full_dataset.csv')
    
    averaged_ratings = compute_average_ratings()
    clustered_utterances, cluster_centers, cluster_emotions = perform_clustering_with_labels(
        averaged_ratings, emotion_df
    )
    
    # Print results
    print("\nCluster Analysis with Emotion Labels:")
    for cluster, data in cluster_emotions.items():
        print(f"\nCluster {cluster}:")
        print(f"Dominant Category: {data['dominant_category']}")
        print(f"Average activation: {data['avg_activation']:.2f}")
        print(f"Average valence: {data['avg_valence']:.2f}")
        print("Category scores:")
        for category, score in data['category_scores'].items():
            print(f"  {category}: {score:.2f}")
        print("Emotion counts:", dict(data['emotion_counts']))