from pydantic import BaseModel
import random
import string
import dspy
import os
from api_keys import TOGETHER_API_KEY, SERPAPI_API_KEY
import util
import numpy as np
from synthetic_users import SYNTHETIC_USERS
from typing import Optional
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import requests
from mem0 import Memory
from datetime import datetime
import time

def greeting(self):
    """Return a message that the chatbot uses to greet the user."""
    ########################################################################
    # TODO: Write a short greeting message                                 #
    ########################################################################

    greeting_message = "Hello! I'm your Movie Ticket Agent. I can recommend movies, book tickets, answer movie questions, and handle special requests. How can I help you today?"

    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return greeting_message

os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

## load ratings matrix and convert user ratings to binary in user_ratings_dict
# note that the users in ratings_matrix are different from the users in user_ratings_dict
titles, ratings_matrix = util.load_ratings('data/ratings.txt')
user_ratings_dict = {user: np.zeros(len(titles)) for user in SYNTHETIC_USERS}
for user, movies in SYNTHETIC_USERS.items():
    for movie in movies:
        user_ratings_dict[user][titles.index(movie)] = 1

class Date(BaseModel):
    # Somehow LLM is bad at specifying `datetime.datetime`, so
    # we define a custom class to represent the date.
    year: int
    month: int
    day: int
    hour: int
    minute: int

class UserProfile(BaseModel):
    name: str
    email: str
    balance: float

class Movie(BaseModel):
    title: str
    start_time: Date
    price: float

class Ticket(BaseModel):
    user_name: str
    movie_title: str
    time: Date

class Request(BaseModel):
    user_request: str
    user_name: str

user_database = {
    "peter": UserProfile(name="Peter", email="peter@gmail.com", balance=42),
    "emma": UserProfile(name="Emma", email="emma@gmail.com", balance=79),
    "jake": UserProfile(name="Jake", email="jake@gmail.com", balance=13),
    "sarah": UserProfile(name="Sarah", email="sarah@gmail.com", balance=36),
    "michael": UserProfile(name="Michael", email="michael@gmail.com", balance=8),
    "lisa": UserProfile(name="Lisa", email="lisa@gmail.com", balance=97),
    "marcus": UserProfile(name="Marcus", email="marcus@gmail.com", balance=59),
    "sophia": UserProfile(name="Sophia", email="sophia@gmail.com", balance=25),
    "chris": UserProfile(name="Chris", email="chris@gmail.com", balance=63),
    "amy": UserProfile(name="Amy", email="amy@gmail.com", balance=91),
}

showtime_database = {
    "Back to the Future": Movie(title="Back to the Future (1985)", start_time=Date(year=2025, month=11, day=13, hour=10, minute=0), price=15.0),
    "Speed": Movie(title="Speed (1994)", start_time=Date(year=2025, month=11, day=13, hour=11, minute=30), price=20.0),
    "Star Wars: Episode VI - Return of the Jedi": Movie(title="Star Wars: Episode VI - Return of the Jedi (1983)", start_time=Date(year=2025, month=11, day=15, hour=13, minute=0), price=18.0),
    "Terminator": Movie(title="Terminator, The (1984)", start_time=Date(year=2025, month=11, day=15, hour=18, minute=0), price=14.0),
    "Star Wars: Episode V - The Empire Strikes Back": Movie(title="Star Wars: Episode V - The Empire Strikes Back (1980)", start_time=Date(year=2025, month=11, day=15, hour=20, minute=0), price=16.5),
    "Matrix": Movie(title="Matrix, The (1999)", start_time=Date(year=2025, month=11, day=15, hour=22, minute=0), price=19.0),
    "Silence of the Lambs": Movie(title="Silence of the Lambs, The (1991)", start_time=Date(year=2025, month=11, day=16, hour=10, minute=15), price=17.0),
    "Fight Club": Movie(title="Fight Club (1999)", start_time=Date(year=2025, month=11, day=16, hour=12, minute=45), price=18.5),
    "Lord of the Rings: The Two Towers": Movie(title="Lord of the Rings: The Two Towers, The (2002)", start_time=Date(year=2025, month=11, day=16, hour=15, minute=0), price=17.5),
    "Lord of the Rings: The Fellowship of the Ring": Movie(title="Lord of the Rings: The Fellowship of the Ring, The (2001)", start_time=Date(year=2025, month=11, day=16, hour=17, minute=30), price=17.0),
    "Pulp Fiction": Movie(title="Pulp Fiction (1994)", start_time=Date(year=2025, month=11, day=16, hour=19, minute=45), price=15.5),
    "Star Wars: Episode IV - A New Hope": Movie(title="Star Wars: Episode IV - A New Hope (1977)", start_time=Date(year=2025, month=11, day=16, hour=22, minute=0), price=16.0),
    "Titanic": Movie(title="Titanic (1997)", start_time=Date(year=2025, month=11, day=15, hour=10, minute=0), price=20.0)
}

ticket_database = {}
request_database = {}


################################################################################################################################################
# PART 1

## defining tools and helper functions for the tools

def _generate_id(length=8):
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))

def similarity(u, v):
    """
    Calculate the cosine similarity between two vectors.
    You may assume that the two arguments have the same shape.
    :param u: one vector, as a 1D numpy array
    :param v: another vector, as a 1D numpy array
    :returns: the cosine similarity between the two vectors
    Note: you should return 0 if u or v has norm 0
    """
    ########################################################################
    # TODO: Compute cosine similarity between the two vectors.             #
    ########################################################################
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0
    similarity = np.dot(u, v) / (norm_u * norm_v)
    ########################################################################
    #                          END OF YOUR CODE                            #
    ########################################################################
    return similarity

def recommend_movies(user_name: str, k=3):
    """
    Generate a list of indices of movies to recommend using collaborative
        filtering.

    You should return a collection of `k` indices of movies recommendations.

    As a precondition, user_ratings have been loaded for you based on the provided user_name.

    Do not recommend movies that the user has already rated (since the ratings of the rated movies are used to calculate similarity to a potential recommendation)
    If the user already rated every movie, then the function should return an empty list.

    Hint: the similarity between two movies is based on the similarity of their ratings in ratings_matrix

    :returns: a list of k movie titles corresponding to movies in
    ratings_matrix, in descending order of recommendation. (the k movie titles correspond to "movie indices" in ratings_matrix)
    """
    user_profile = user_database[user_name.lower()]
    user_name = user_profile.name
    user_ratings = user_ratings_dict[user_name]

    ########################################################################
    # TODO: Implement collaborative filtering to generate a list of movie  #
    # indices to recommend to the user.                                    #
    ########################################################################
    # Populate this list with k movie indices to recommend to the user.
    rated_indices = [i for i in range(len(user_ratings)) if user_ratings[i] != 0]
    unrated_indices = [i for i in range(len(user_ratings)) if user_ratings[i] == 0]

    if not unrated_indices:
        return []

    scores = []
    for movie_idx in unrated_indices:
        score = 0.0
        for rated_idx in rated_indices:
            score += user_ratings[rated_idx] * similarity(ratings_matrix[movie_idx], ratings_matrix[rated_idx])
        scores.append((movie_idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    recommendations = [idx for idx, score in scores[:k]]

    ########################################################################
    #                          END OF YOUR CODE                            #
    ########################################################################

    ## convert the movie indices to movie titles
    result_titles = [titles[movie_index] for movie_index in recommendations]
    return result_titles

def general_qa(user_request: str):
    """
    Answer a general question about movies by making an LLM call.
    """
    lm = dspy.LM("together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1")
    dspy.configure(lm=lm)

    response = lm(messages=[{"role": "user", "content": user_request}])
    return response

def find_time(movie_title: str):
    """
    Find the time of the given movie title. Title must be one of: 
    [Back to the Future, Speed, Star Wars: Episode VI - Return of the Jedi, 
    Terminator, Star Wars: Episode V - The Empire Strikes Back, Matrix, 
    Silence of the Lambs, Fight Club, Lord of the Rings: The Two Towers, 
    Lord of the Rings: The Fellowship of the Ring, Pulp Fiction, 
    Star Wars: Episode IV - A New Hope, Titanic]
    """
    movie = showtime_database[movie_title]
    return movie.start_time

def find_price(movie_title: str):
    """
    Find the price of the given movie title. Title must be one of: 
    [Back to the Future, Speed, Star Wars: Episode VI - Return of the Jedi, 
    Terminator, Star Wars: Episode V - The Empire Strikes Back, Matrix, 
    Silence of the Lambs, Fight Club, Lord of the Rings: The Two Towers, 
    Lord of the Rings: The Fellowship of the Ring, Pulp Fiction, 
    Star Wars: Episode IV - A New Hope, Titanic]
    """
    movie = showtime_database[movie_title]
    return movie.price

def find_balance(user_name: str):
    """
    Find the balance of the given user name. 
    Name must be one of: [peter, emma, jake, sarah, michael, lisa, marcus, sophia, chris, amy]
    """
    user_profile = user_database[user_name.lower()]
    return user_profile.balance

def file_request(user_request: str, user_name: str = ""):
    """
    File a human customer support request if this is something the agent cannot handle.
    """
    request_id = _generate_id(length=6)
    request_database[request_id] = Request(
        user_request=user_request,
        user_name=user_name,
    )
    return request_id


def book_ticket(user_name: str, movie_title: str):
    """
    Book a ticket for the given user and movie title. Tile must be one of: 
    [Back to the Future, Speed, Star Wars: Episode VI - Return of the Jedi, 
    Terminator, Star Wars: Episode V - The Empire Strikes Back, Matrix, 
    Silence of the Lambs, Fight Club, Lord of the Rings: The Two Towers, 
    Lord of the Rings: The Fellowship of the Ring, Pulp Fiction, 
    Star Wars: Episode IV - A New Hope, Titanic]
    """
   
    ########################################################################
    # TODO: Implement the `book_ticket` tool
    # * Only make a booking if the user has enough balance. Then, update the
    #   user's balance in `ticket_database`.
    #  If there is not enough balance, return: "Insufficient balance to book the ticket for {movie_title}."
    # * Use `_generate_id` to create a 6-digit ticket number for the booking
    # * For any requests that can't be handled by your agent, make a human
    #   customer support request by calling the `file_request` tool
    #   to add the request to the `request_database`
    ########################################################################
    user_profile = user_database[user_name.lower()]
    movie = showtime_database[movie_title]

    if user_profile.balance < movie.price:
        return f"Insufficient balance to book the ticket for {movie_title}. Current balance: ${user_profile.balance:.2f}, ticket price: ${movie.price:.2f}."

    ticket_number = _generate_id(length=6)
    user_profile.balance -= movie.price
    user_balance = user_profile.balance

    ticket_database[ticket_number] = Ticket(
        user_name=user_profile.name,
        movie_title=movie_title,
        time=movie.start_time,
    )

    print(f"\nPrinting ticket_database:")
    import pprint as _pprint
    _pprint.pprint(ticket_database)
    print()
    ########################################################################
    #                          END OF YOUR CODE                            #
    ########################################################################
    return f"Ticket booked successfully for {user_name} for the movie {movie_title}. The ticket number is {ticket_number}. Your new balance is {user_balance}."


## Integrating tools into an LLM agent: you will use the agent below for part 1

# The MovieTicketAgent class is a wrapper that modifies dspy.Signature. If you are curious
# about the signature, read the documentation here:
# https://dspy.ai/learn/programming/signatures/#class-based-dspy-signatures

class MovieTicketAgent(dspy.Signature):
    ########################################################################
    ## TODO: Add a few sentences to flesh out the agent objective in the docstring below.
    # In DSPy, the docstring of a Signature acts as the system prompt
    # for the language model. It defines the agent’s role, constraints,
    # and decision-making strategy. So it is crucial to define it well!
    # Hint: you can add details about what tools the agent will need to call
    # in order to successfully complete the tasks
    ########################################################################
    """
    You are a movie ticket agent that helps users book and manage movie tickets. You are given a list of tools to handle user requests, and you should decide the right tool to use in order to fulfill users’ requests.

    Tool selection guide:
    - recommend_movies: user asks for movie recommendations (requires user_name and k)
    - general_qa: static movie knowledge — plot summaries, classic cast/crew, film history
    - web_search: use for ANY question about actors, directors, current cast, who starred in a film, who directed a film, or recent movie trivia. Do NOT use general_qa for these — use web_search instead. Examples: "Who played X in Y?", "Who directed Z?", "What is the cast of ...?"
    - find_time: look up showtime for a movie
    - find_price: get ticket price for a movie
    - find_balance: check a user’s account balance
    - book_ticket: purchase a ticket (check balance first; inform user if insufficient)
    - file_request: log requests you cannot handle (discounts, refunds, special accommodations)

    When memory tools are available:
    - store_memory: user asks you to remember something about them
    - search_memories: ALWAYS call this FIRST when a user asks about their own preferences, history, or anything they told you before (e.g., "What is my favorite movie?", "What did I tell you?")
    - web_search: user explicitly asks to search the web, OR asks about actors/directors/current cast

    Examples of correct tool routing:
    - "Who played Lucy in Materialists?" → web_search (current cast info)
    - "Who directed The Matrix?" → web_search (director info)
    - "What is the plot of Inception?" → general_qa (plot summary)
    - "Book a ticket for Matrix for Peter" → book_ticket
    - "Recommend movies for emma" → recommend_movies
    - "Remember I love sci-fi movies" → store_memory
    - "What is my favorite genre?" → search_memories (recall stored preference)

    Always be helpful and concise. When booking tickets, confirm the ticket number and the user’s updated balance. When making recommendations, provide the list of movie titles returned by the tool. For unhandled requests, acknowledge the limitation and confirm the support request has been filed.
    """
    ########################################################################
    #                          END OF YOUR CODE                            #
    ########################################################################
    
    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc="Message that summarizes the process result, and the information users need, e.g., the ticket number if a new ticket is booked."
    )

dspy.configure(lm=dspy.LM("together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"))

react_agent = dspy.ReAct(
    MovieTicketAgent,
    tools = [
        recommend_movies,
        general_qa,
        ########################################################################
        ## TODO: add other tools for your agent here
        ########################################################################
        book_ticket,
        find_time,
        find_price,
        find_balance,
        file_request,
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
    ]
)


################################################################################################################################################
# PART 2

## Part 2: web search utilities 

def extract_text(html: str) -> str:
    """
    Extracts clean, readable text from raw HTML.

    This function takes an HTML page (returned from a web request) and removes
    non-readable content. 

    Args:
        html (str): Raw HTML content of a web page.

    Returns:
        str: Cleaned plain-text version of the page content.
    """
    soup = BeautifulSoup(html, "html.parser")   # Parse the raw HTML into a structured BeautifulSoup object
    # Remove tags that do not contain meaningful visible text (scripts, styles, ads, nav, etc.)
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                     "aside", "form", "button", "iframe", "advertisement", "banner"]):
        tag.decompose()
    # Also remove elements with ad/nav-related class or id attributes
    for tag in soup.find_all(True):
        cls = " ".join(tag.get("class") or [])
        tid = str(tag.get("id") or "")
        if any(kw in cls.lower() or kw in tid.lower() for kw in ("nav", "menu", "ad", "banner", "sidebar", "cookie", "popup")):
            tag.decompose()
    text = soup.get_text(separator=" ", strip=True)  # Extract all remaining visible text from the HTML
    return " ".join(text.split())


class WebTools:
    """
    Utility class that provides web search and page-reading tools for the agent.

    This class acts as a thin wrapper around a web search API (SerpAPI).
    The agent calls these methods when it needs *external information*
    that is not available in its prompt or memory.

    Conceptually:
    - The LLM decides *when* to search
    - This class handles *how* the search is executed
    - The returned text is formatted to be readable by both humans and LLMs
    """

    def __init__(self, serpapi_key: Optional[str] = None):
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")

    def _fetch_page_text(self, url: str, max_chars: int = 1500) -> str:
        """Fetch a URL and return cleaned plain text, truncated to max_chars."""
        try:
            resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                return extract_text(resp.text)[:max_chars]
        except Exception:
            pass
        return ""

    def web_search(self, query: str, num_results: int = 5, page: int = 1) -> str:
        """
        Search the web and return top links/snippets.
        Args:
          query: search query string
          num_results: number of results to return (recommended <= 10)
          page: pagination index starting from 1
        Returns:
          A formatted list of results with title, link, and snippet.
        """
        if not self.serpapi_key:
            return "Error: SERPAPI_API_KEY is not set."

        # Google via SerpAPI. Pagination is controlled by 'start' (0-indexed).
        # page=1 => start=0, page=2 => start=num_results, etc.
        start = (max(page, 1) - 1) * num_results
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "num": num_results,
            "start": start,
        }

        try:
            results = GoogleSearch(params).get_dict()  # Execute the search request and parse the JSON response
            organic = results.get("organic_results", []) or []
            if not organic:
                return "No results found."
            # Build a human- and LLM-readable summary of the results
            lines = [f"Web search results for: {query} (page {page})"]
            for i, item in enumerate(organic[:num_results], 1):
                title = item.get("title") or "(no title)"
                link = item.get("link") or "(no link)"
                snippet = item.get("snippet") or ""
                # Each result is formatted as a numbered block
                lines.append(f"{i}. {title}\n   {link}\n   {snippet}".strip())
                # Fetch full page text for the top 2 results to give the LLM more context
                if i <= 2 and link != "(no link)":
                    page_text = self._fetch_page_text(link)
                    if page_text:
                        lines.append(f"   [Page content]: {page_text}")

            return "\n".join(lines)

        except Exception as e:
            return f"Error during web_search: {str(e)}"



# memory functionalities
## Memory configuration
memory_config = {
    "llm": {
        "provider": "together",
        "config": {
            "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "together",
        "config": {
            "model": "intfloat/multilingual-e5-large-instruct"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "embedding_model_dims": 1024
        }
    }
}


## Part 2: memory utilities
class MemoryTools:
    """Tools for interacting with the Mem0 memory system."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def store_memory(self, content: str, user_id: str = "default_user") -> str:
        """
        Store a piece of information in memory.

        This is typically called when the agent learns something that should
        persist across turns (e.g., user preferences, reminders, personal facts).

        Args:
            content (str): The text to store in memory.
            user_id (str): Identifier for the user whose memory this belongs to.

        Returns:
            str: A confirmation message or an error message.
        """
        try:
            ########################################################################
            # TODO: add the content to Mem0's memory store for user_id
            # Hint: It may be helpful to review mem0's memory operations here:
            # https://docs.mem0.ai/core-concepts/memory-operations
            ########################################################################
            self.memory.add(content, user_id=user_id)
            ########################################################################
            #                          END OF YOUR CODE                            #
            ########################################################################
            return f"Stored memory: {content}"
        except Exception as e:
            return f"Error storing memory: {str(e)}"

    def create_memory(self, results):
        """
        Helper function that creates a memory_text string containing all of the memories. You will call this function
        in search_memories and get_all_memories().
        This function should return memory_text
        """
        memory_text = "Relevant memories found:\n"
        for i, result in enumerate(results["results"]):
            memory_text += f"{i}. {result['memory']}\n"
        return memory_text

    def search_memories(self, query: str, user_id: str = "default_user", limit: int = 5) -> str:
        """
        Search memory for items relevant to a query.

        This is used when the agent needs to recall previously stored information,
        such as user preferences or earlier statements.

        Args:
            query (str): Natural-language search query.
            user_id (str): Identifier for the user whose memory should be searched.
            limit (int): Maximum number of memories to return.

        Returns:
            str: A formatted list of relevant memories or a message indicating
            that nothing was found.
        """
        try:
            ########################################################################
            # TODO: search for relevant memories and store them in results
            # Hint: it would be helpful to read the documentation of
            # mem0 to see how to use the `search` method: https://github.com/mem0ai/mem0
            ########################################################################
            results = self.memory.search(query, user_id=user_id, limit=limit)
            ########################################################################
            #                          END OF YOUR CODE                            #
            ########################################################################
            if not results:
                return "No relevant memories found."
            memory_text = self.create_memory(results)
            return memory_text
        except Exception as e:
            return f"Error searching memories: {str(e)}"

    def get_all_memories(self, user_id: str = "default_user") -> str:
        """Get all memories for a user."""
        try:
            results = self.memory.get_all(user_id=user_id)
            if not results:
                return "No memories found for this user."

            memory_text = self.create_memory(results)
            return memory_text
        except Exception as e:
            return f"Error retrieving memories: {str(e)}"
    
    def update_memory(self, memory_id: str, new_content: str) -> str:
        """Update an existing memory."""
        try:
            ########################################################################
            # TODO: Replace the old memory content with the new content
            # Hint: It may be helpful to review mem0's memory operations here:
            # https://docs.mem0.ai/core-concepts/memory-operations
            ########################################################################
            self.memory.update(memory_id, data=new_content)
            ########################################################################
            #                          END OF YOUR CODE                            #
            ########################################################################
            return f"Updated memory with new content: {new_content}"
        except Exception as e:
            return f"Error updating memory: {str(e)}"

    def delete_memory(self, memory_id: str) -> str:
        """Delete a specific memory."""
        try:
            ########################################################################
            # TODO: delete the memory for a given memory_id
            # Hint: It may be helpful to review mem0's memory operations here:
            # https://docs.mem0.ai/core-concepts/memory-operations
            ########################################################################
            self.memory.delete(memory_id)
            ########################################################################
            #                          END OF YOUR CODE                            #
            ########################################################################
            return "Memory deleted successfully."
        except Exception as e:
            return f"Error deleting memory: {str(e)}"


# other helper functions
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_reminder(reminder_text: str, date_time: str = None, user_id: str = "default_user") -> str:
    """Set a reminder for the user."""
    reminder = f"Reminder set for {date_time}: {reminder_text}"
    # This will be connected to memory_tools in the agent
    return reminder


def get_preferences(category: str = "general", user_id: str = "default_user") -> str:
    """Get user preferences for a specific category."""
    # This will be connected to memory_tools in the agent
    return f"Getting preferences for {category}"


def update_preferences(category: str, preference: str, user_id: str = "default_user") -> str:
    """Update user preferences."""
    # This will be connected to memory_tools in the agent
    return f"Updated {category} preference to {preference}"


## You will use the enhanced agent below for part 2
class EnhancedMovieTicketAgent(dspy.Module):
    """Movie ticket agent with web search and memory capabilities.  You have access to web search to find current movie information, memory to remember user preferences,
    and various tools to handle user requests. You should decide the right tool to use in order to
    fulfill users' request.
    
    When users share preferences or information, store it in memory.
    When you need to recall user preferences, search memories.
    When you need current movie information, use web search."""

    def __init__(self, enable_web_search=True, enable_memory=True):
        super().__init__()
        
        # Initialize web tools
        self.web_tools = WebTools() if enable_web_search else None
        
        # Initialize memory
        if enable_memory:
            self.memory = Memory.from_config(memory_config)
            self.memory_tools = MemoryTools(self.memory)
        else:
            self.memory = None
            self.memory_tools = None
        
        ########################################################################
        # TODO: Add tools for the base agent, as well as web search and memory
        # if they are enabled
        ########################################################################
        # TODO: Add tools for the base agent
        self.tools = [
            recommend_movies,
            general_qa,
            book_ticket,
            find_time,
            find_price,
            find_balance,
            file_request,
        ]

        # enable web search
        if self.web_tools:
            self.tools.append(self.web_tools.web_search)

        # add memory tools if enabled
        if self.memory_tools:
            self.tools.extend([
                self.memory_tools.store_memory,
                self.memory_tools.search_memories,
                self.memory_tools.get_all_memories,
                self.memory_tools.update_memory,
                self.memory_tools.delete_memory,
            ])

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        # Initialize ReAct agent
        self.react = dspy.ReAct(
            MovieTicketAgent,
            tools=self.tools,
            max_iters=6
        )
    
    def forward(self, user_request: str):
        """Process user input with enhanced capabilities."""
        return self.react(user_request=user_request)


try:
    enhanced_agent = EnhancedMovieTicketAgent(enable_web_search=True, enable_memory=True)
except Exception as e:
    print(f"Warning: could not initialize enhanced_agent: {e}")
    enhanced_agent = None
