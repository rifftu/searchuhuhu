# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def _graphSearch(problem: SearchProblem, fringe: util.Stack):

    """
    Fringe items are tuples of form (path, node) where
    path is a list of directions e.g. ["North", "West"]
    node is the coords of the current node e.g. (5,4)
    cost is how much it took to get here
    h is the heuristic (a* only)

    closed set items are tuples that are the coords of nodes
    """

    closed = set()

    fringe.push(([], problem.getStartState()))

    while not fringe.isEmpty():

        curr_path, curr_state = fringe.pop()

        if problem.isGoalState(curr_state):
            return curr_path

        if curr_state not in closed:
            closed.add(curr_state)
            for child in problem.getSuccessors(curr_state):

                fringe.push((curr_path + [child[1]], child[0]))

    return 0



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    return _graphSearch(problem, util.Stack())

4
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    return _graphSearch(problem, util.Queue())

4
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return _aStar(problem, nullHeuristic, util.PriorityQueue())

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return _aStar(problem, heuristic, util.PriorityQueue())

def _aStar(problem: SearchProblem, heuristic, fringe: util.PriorityQueue):

    def get_path(state, start):
    # Takes the final backstep dictionary and steps back until it has the optimal path
        path = []

        while state != start and backstep:

            step = backstep.pop(state)

            path = [step[2]] + path
            state = step[1]

        return path


    closed = set()
    # set of expanded nodes (not to visit again)

    backstep = {}
    # Keeps track of previous step to each node & best total cost so far
    # Format: {state: (total_cost, parent_state, prev_direction)}

    # Add start node to both backstep and fringe
    fringe.push(problem.getStartState(), 0)
    backstep[problem.getStartState()] = (0, None, None)

    while not fringe.isEmpty():

        curr_state = fringe.pop()
        curr_cost = backstep[curr_state][0]

        # Check for goal state
        if problem.isGoalState(curr_state):
            return get_path(curr_state, problem.getStartState())

        if curr_state not in closed:
            # Each node expanded at most once
            closed.add(curr_state)

            for child in problem.getSuccessors(curr_state):
                # Do the expansion

                # Obvious
                if child[0] in closed:
                    continue

                # Add cost to child to get current optimal distance to child
                child_cost = curr_cost + child[2]

                # If there is already a better way to the child, this is false
                if (child[0] not in backstep) or (child_cost < backstep[child[0]][0]):

                    backstep[child[0]] = (child_cost, curr_state, child[1])
                    # Adds cost, parent, and direction to backstep

                    fringe.update(child[0], child_cost + heuristic(child[0], problem))
                    # Adds child node and priority to the PQ

    return 0


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
