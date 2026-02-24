import numpy as np
import sys
import argparse
import unittest.mock as mock
sys.path.append('.')
import agent
from agent import similarity, recommend_movies

def test_similarity():
    # Identical vectors → similarity of 1
    u = np.array([1.0, 2.0, 3.0])
    result = similarity(u, u)
    assert np.isclose(result, 1.0), f"Identical vectors test failed: {result}"

    # Opposite vectors → similarity of -1
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([-1.0, 0.0, 0.0])
    result = similarity(u, v)
    assert np.isclose(result, -1.0), f"Opposite vectors test failed: {result}"

    # Orthogonal vectors → similarity of 0
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    result = similarity(u, v)
    assert np.isclose(result, 0.0), f"Orthogonal vectors test failed: {result}"

    # Known value: [1,1] and [1,0] → cos(45°) = 1/sqrt(2)
    u = np.array([1.0, 1.0])
    v = np.array([1.0, 0.0])
    result = similarity(u, v)
    assert np.isclose(result, 1.0 / np.sqrt(2)), f"Known value test failed: {result}"

    # Zero vector → should return 0 (not crash)
    u = np.array([0.0, 0.0, 0.0])
    v = np.array([1.0, 2.0, 3.0])
    result = similarity(u, v)
    assert result == 0, f"Zero vector test failed: {result}"

    # Symmetry: similarity(u, v) == similarity(v, u)
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    assert np.isclose(similarity(u, v), similarity(v, u)), "Symmetry test failed"

    # Result should always be in [-1, 1]
    u = np.array([3.0, -1.0, 2.0])
    v = np.array([-2.0, 4.0, 1.0])
    result = similarity(u, v)
    assert -1.0 <= result <= 1.0, f"Out of range result: {result}"

    # Different-length vectors → should raise ValueError
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 2.0])
    try:
        result = similarity(u, v)
        assert False, "Different-length vectors test failed: expected ValueError but got result"
    except ValueError:
        pass  # expected

    print("All similarity tests passed!")



def test_recommend_movies():
    # --- mock data setup ---
    # 4 movies, each represented as a row vector of "features"
    mock_ratings_matrix = np.array([
        [1.0,  0.0,  0.0],   # movie 0
        [1.0,  0.1,  0.0],   # movie 1 — very similar to movie 0
        [0.0,  0.0,  1.0],   # movie 2 — different
        [0.0,  0.0,  1.1],   # movie 3 — very similar to movie 2
    ])
    mock_titles = ["Movie A", "Movie B", "Movie C", "Movie D"]

    # user has rated movie 0 positively and movie 2 negatively
    # so we expect movie 1 recommended (similar to 0) before movie 3 (similar to 2)
    mock_user_ratings = [1.0, 0.0, -1.0, 0.0]

    class FakeProfile:
        name = "Alice"

    mock_user_database = {"alice": FakeProfile()}
    mock_user_ratings_dict = {"Alice": mock_user_ratings}

    with mock.patch.multiple(
        agent,
        user_database=mock_user_database,
        user_ratings_dict=mock_user_ratings_dict,
        ratings_matrix=mock_ratings_matrix,
        titles=mock_titles,
    ):
        # Basic: returns k results
        result = recommend_movies("alice", k=2)
        assert len(result) == 2, f"Expected 2 recommendations, got {len(result)}: {result}"

        # Should not recommend already-rated movies (movie 0 and movie 2)
        assert "Movie A" not in result, "Recommended an already-rated movie (Movie A)"
        assert "Movie C" not in result, "Recommended an already-rated movie (Movie C)"

        # Movie B (similar to liked Movie A) should rank above Movie D (similar to disliked Movie C)
        assert result.index("Movie B") < result.index("Movie D"), \
            f"Expected Movie B before Movie D, got: {result}"

        # k=1 returns exactly 1 result
        result = recommend_movies("alice", k=1)
        assert len(result) == 1, f"Expected 1 recommendation, got {len(result)}"
        assert result[0] == "Movie B", f"Expected Movie B as top recommendation, got {result[0]}"

        # All movies rated: no recommendations possible
        mock_user_ratings_all_rated = [1.0, -1.0, -1.0, 1.0]
        mock_user_ratings_dict_full = {"Alice": mock_user_ratings_all_rated}
        with mock.patch.object(agent, 'user_ratings_dict', mock_user_ratings_dict_full):
            result = recommend_movies("alice", k=3)
            assert result == [], f"Expected no recommendations when all movies rated, got {result}"

        # k greater than number of unrated movies: returns only what's available (Movie B and Movie D)
        result = recommend_movies("alice", k=100)
        assert len(result) == 2, f"Expected 2 recommendations when k > available movies, got {len(result)}"
        assert "Movie A" not in result, "Recommended an already-rated movie (Movie A) in large-k test"
        assert "Movie C" not in result, "Recommended an already-rated movie (Movie C) in large-k test"

    print("All recommend_movies tests passed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', choices=['similarity', 'recommend_movies'], required=True,
                        help='Which function to test')
    args = parser.parse_args()
    if args.function == 'similarity':
        test_similarity()
    elif args.function == 'recommend_movies':
        test_recommend_movies()