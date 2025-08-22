from scipy.optimize import linear_sum_assignment
import numpy as np

class Track:
    def __init__(self, track_id, pos):
        self.id = track_id
        self.pos = pos


def associate_tracks(tracks1, tracks2, pseudo_value=None):
    if not tracks1:
        return []  # Return an empty list if one or both sets are empty
    elif len(tracks1) == len(tracks2):
        return associate_tracks_by_x_order(tracks1, tracks2)
    else:
        associations = associate_tracks_by_horizontal_distance(tracks1, tracks2)
        # Check if any tracks are unassigned in tracks2
        if len(associations) < len(tracks1):
            # Create pseudo associations for unassigned tracks in tracks2
            assigned_tracks = set(track1 for track1, _ in associations.items())
            unassigned_tracks = [track1 for track1 in tracks1 if track1 not in assigned_tracks]
            if pseudo_value is None:
                pseudo_value = Track(-1, (-1, -1))  # Placeholder track with specific values
            for track in unassigned_tracks:
                associations[track] = pseudo_value
        elif len(associations) < len(tracks2):
            # Create pseudo associations for unassigned tracks in tracks2
            assigned_tracks = set(track2 for _, track2 in associations.items())
            unassigned_tracks = [track2 for track2 in tracks2 if track2 not in assigned_tracks]
            if pseudo_value is None:
                pseudo_value = Track(-2, (-2, -2))  # Placeholder track with specific values
            for track in unassigned_tracks:
                associations[track] = pseudo_value
        return associations

def associate_tracks_by_x_order(tracks1, tracks2):
    sorted_tracks1 = sorted(tracks1, key=lambda t: t.pos[0])
    sorted_tracks2 = sorted(tracks2, key=lambda t: t.pos[0])

    if len(sorted_tracks1) != len(sorted_tracks2):
        raise ValueError("The number of tracks in both sets must be the same.")

    associations = {}

    for i in range(len(sorted_tracks1)):
        associations[sorted_tracks1[i]] = sorted_tracks2[i]

    return associations


def associate_tracks_by_horizontal_distance(tracks1, tracks2):
    # Create a distance matrix to store distances between tracks1 and tracks2
    distances = np.zeros((len(tracks1), len(tracks2)))

    # Calculate distances between all pairs of tracks
    for i, track1 in enumerate(tracks1):
        for j, track2 in enumerate(tracks2):
            distances[i, j] = abs(track1.pos[0] - track2.pos[0])

    # If all distances are infinity, no associations can be made
    if np.all(np.isinf(distances)):
        return {}

    # Initial associations based on minimum distances
    row_indices, col_indices = linear_sum_assignment(distances)
    associations = {tracks1[i]: tracks2[j] for i, j in zip(row_indices, col_indices)}

    return associations


# Example usage:
tracks1 = [Track(1, (1097.76, 581.70)), Track(2, (1155.14, 1017.72)), Track(3, (857.4, 1159.88))]
tracks2 = [Track(4, (1087.30, 951.70)), Track(5, (1129.81, 1122.32)), Track(6, (707.98, 976.09))]

associations = associate_tracks(tracks1, tracks2)
for track1, track2 in associations.items():
    print(f"Track {track1.id} is associated with Track {track2.id}")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

tracks3 = [Track(1, (1097.76, 581.70)), Track(2, (1155.14, 1017.72)), Track(3, (857.4, 1159.88))]
tracks4 = [Track(4, (1087.30, 951.70)), Track(5, (1129.81, 1122.32)), Track(6, (707.98, 976.09)), Track(7, (1071.96, 437.02))]

associations2 = associate_tracks(tracks3, tracks4)
for track1, track2 in associations2.items():
    print(f"Track {track1.id} is associated with Track {track2.id}")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

tracks5 = [Track(1, (1097.76, 581.70)), Track(2, (1155.14, 1017.72)), Track(3, (857.4, 1159.88)),
           Track(7, (1071.96, 437.02)), Track(8, (917.19, 980.75))]
tracks6 = [Track(4, (1087.30, 951.70)), Track(5, (1129.81, 1122.32)), Track(6, (707.98, 976.09))]

associations3 = associate_tracks(tracks5, tracks6)
for track1, track2 in associations3.items():
    print(f"Track {track1.id} is associated with Track {track2.id}")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
