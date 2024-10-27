from collections import Counter
import os
import re

import pandas as pd


def parse_emotion_file(content):
    """Parse emotion labels from file content"""
    utterances = {}
    for line in content.split("\n"):
        if not line.strip():
            continue
        # Split line into utterance ID and emotions
        parts = line.split(" :")
        if len(parts) < 2:
            continue

        utterance_id = parts[0].strip()
        # Extract all emotion labels
        emotions = re.findall(r":(\w+);", line)
        if 'Neutral state' in line:
            emotions += ['Neutral state']

        # Map similar emotions
        mapped_emotions = []
        for emotion in emotions:
            if emotion in ["Happiness", "Excited"]:
                mapped_emotions.append("happy")
            elif emotion == 'Neutral state':
                mapped_emotions.append('neutral')
            elif emotion in ["Sadness"]:
                mapped_emotions.append("sad")
            elif emotion in ["Anger"]:
                mapped_emotions.append("angry")
            # Ignore other emotions for this analysis

        if mapped_emotions:
            utterances[utterance_id] = mapped_emotions

    return utterances


def get_majority_vote(evaluations):
    """Determine majority vote from multiple evaluations"""
    # Count occurrences of each emotion
    emotion_counts = Counter()
    for emotions in evaluations:
        for emotion in emotions:
            emotion_counts[emotion] += 1

    # Find emotion with highest count
    if not emotion_counts:
        return None

    max_count = max(emotion_counts.values())
    max_emotions = [e for e, c in emotion_counts.items() if c == max_count]

    # Return majority emotion only if it's unique and has more than 1 vote
    if len(max_emotions) == 1 and max_count > 1:
        return max_emotions[0]
    return None


def analyze_majority_votes():
    """Analyze majority votes across multiple evaluator files"""
    # Parse each evaluator file
    evaluations_by_utterance = {}

    for session in range(1, 6):
        folderpath = f"Processed/IEMOCAP_full_release/Session{session}/dialog/EmoEvaluation/Categorical/"
        evaluator_files = [
            folderpath + filename for filename in os.listdir(folderpath) if filename
        ]

        for file in evaluator_files:
            try:
                with open(file, "r") as file:
                    content = file.read()
            except FileNotFoundError:
                print(f"Warning: Could not find file {file}")

            utterance_emotions = parse_emotion_file(content)

            # Group evaluations by utterance
            for utterance_id, emotions in utterance_emotions.items():
                if utterance_id not in evaluations_by_utterance:
                    evaluations_by_utterance[utterance_id] = []
                evaluations_by_utterance[utterance_id].append(emotions)

    # Get majority votes
    majority_votes = {}
    for utterance_id, evaluations in evaluations_by_utterance.items():
        majority = get_majority_vote(evaluations)
        majority_votes[utterance_id] = majority

    return majority_votes, evaluations_by_utterance


# Example usage
if __name__ == "__main__":

    majority_votes, _ = analyze_majority_votes()

    # Print summary statistics
    emotion_counts = Counter(majority_votes.values())
    print("\nEmotion distribution in majority votes:")
    for emotion, count in emotion_counts.most_common():
        print(f"{emotion}: {count}")

    df = pd.DataFrame(list(majority_votes.items()), columns=["utterance", "label"])
    df.set_index('utterance', inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(
        "Processed/clustered_labels.csv",
    )

    print(f"\nTotal utterances with majority vote: {df.notnull().all(axis=1).sum()}")
    print(f"Total utterances processed: {len(df)}")
    print(
        f"Percentage with majority vote: {df.notnull().all(axis=1).sum()/len(df)*100:.1f}%"
    )

    # # Print example results for each session
    # print("\nExample majority votes by session:")
    # sessions = {}
    # for utterance in majority_votes:
    #     session = utterance.split('_')[0] + '_' + utterance.split('_')[1]
    #     if session not in sessions:
    #         sessions[session] = []
    #     sessions[session].append(utterance)

    # for session in sorted(sessions):
    #     print(f"\n{session}:")
    #     for utterance in sorted(sessions[session])[:5]:  # Show first 5 examples
    #         print(f"{utterance}: {majority_votes[utterance]}")
