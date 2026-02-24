"""
Synthetic user profiles for testing movie recommendations.
Each user has a list of movies they like (actual titles from data/movies.txt).
"""

# Dictionary mapping user names to lists of movies they like
SYNTHETIC_USERS = {
    "Peter": [
        # Sci-Fi fan
        "Matrix, The (1999)",
        "Star Wars: Episode IV - A New Hope (1977)",
        "Star Wars: Episode V - The Empire Strikes Back (1980)",
        "Inception (2010)",
        "Blade Runner (1982)",
        "Aliens (1986)",
        "Alien (1979)",
        "Terminator, The (1984)",
        "Terminator 2: Judgment Day (1991)",
        "Jurassic Park (1993)",
        "Twelve Monkeys (a.k.a. 12 Monkeys) (1995)",
    ],
    
    "Emma": [
        # Romance fan
        "Titanic (1997)",
        "Notebook, The (2004)",
        "Sense and Sensibility (1995)",
        "Bridges of Madison County, The (1995)",
        "Forrest Gump (1994)",
        "When Harry Met Sally... (1989)",
        "Sleepless in Seattle (1993)",
        "Pretty Woman (1990)",
        "Ghost (1990)",
    ],
    
    "Jake": [
        # Action fan
        "Die Hard (1988)",
        "Die Hard: With a Vengeance (1995)",
        "Matrix, The (1999)",
        "Terminator 2: Judgment Day (1991)",
        "Heat (1995)",
        "Gladiator (2000)",
        "Batman Begins (2005)",
        "Spider-Man (2002)",
        "Iron Man (2008)",
        "Casino (1995)",
    ],
    
    "Sarah": [
        # Fantasy/Adventure fan
        "Lord of the Rings: The Fellowship of the Ring, The (2001)",
        "Lord of the Rings: The Two Towers, The (2002)",
        "Lord of the Rings: The Return of the King, The (2003)",
        "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
        "Harry Potter and the Chamber of Secrets (2002)",
        "Star Wars: Episode IV - A New Hope (1977)",
        "Princess Bride, The (1987)",
        "Willow (1988)",
        "NeverEnding Story, The (1984)",
    ],
    
    "Michael": [
        # Comedy fan
        "Pulp Fiction (1994)",
        "Forrest Gump (1994)",
        "Big Lebowski, The (1998)",
        "Groundhog Day (1993)",
        "Shawshank Redemption, The (1994)",
        "Truman Show, The (1998)",
        "American Beauty (1999)",
        "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)",
        "Good Will Hunting (1997)",
    ],
    
    "Lisa": [
        # Animation/Family fan
        "Toy Story (1995)",
        "Finding Nemo (2003)",
        "Incredibles, The (2004)",
        "Shrek (2001)",
        "Monsters, Inc. (2001)",
        "Lion King, The (1994)",
        "Beauty and the Beast (1991)",
        "Aladdin (1992)",
        "Up (2009)",
        "WALL·E (2008)",
    ],
    
    "Marcus": [
        # Horror/Thriller fan
        "Silence of the Lambs, The (1991)",
        "Seven (a.k.a. Se7en) (1995)",
        "Alien (1979)",
        "Shining, The (1980)",
        "Exorcist, The (1973)",
        "Sixth Sense, The (1999)",
        "Memento (2000)",
        "Psycho (1960)",
    ],
    
    "Sophia": [
        # Drama fan
        "Godfather, The (1972)",
        "Godfather: Part II, The (1974)",
        "Shawshank Redemption, The (1994)",
        "Schindler's List (1993)",
        "Fight Club (1999)",
        "Good Will Hunting (1997)",
        "American Beauty (1999)",
        "Requiem for a Dream (2000)",
        "Eternal Sunshine of the Spotless Mind (2004)",
    ],
    
    "Chris": [
        # Eclectic/Popular movies
        "Pulp Fiction (1994)",
        "Matrix, The (1999)",
        "Inception (2010)",
        "Fight Club (1999)",
        "Forrest Gump (1994)",
        "Lord of the Rings: The Fellowship of the Ring, The (2001)",
        "Star Wars: Episode V - The Empire Strikes Back (1980)",
        "Goodfellas (1990)",
        "Silence of the Lambs, The (1991)",
    ],
    
    "Amy": [
        # Rom-Com fan
        "When Harry Met Sally... (1989)",
        "Sleepless in Seattle (1993)",
        "You've Got Mail (1998)",
        "Notting Hill (1999)",
        "Bridget Jones's Diary (2001)",
        "Love Actually (2003)",
        "Clueless (1995)",
        "(500) Days of Summer (2009)",
    ],
}
