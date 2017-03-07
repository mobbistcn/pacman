# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
	
	dist = manhattanDistance(ghostState.getPosition(), newPos)
	score = dist + successorGameState.getScore()
	closest = 100
	for food in newFood.asList():
		fooddist = util.manhattanDistance(food, newPos)
		closest = min(closest, fooddist)
	score -=  closest
	return score
        

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def miniMax(self, gameState, depth, agentIndex):

    actions = []
    if gameState.isWin() or gameState.isLose() or depth == 0:
      return ( self.evaluationFunction(gameState), Directions.STOP)
    numAgents = gameState.getNumAgents()
    if agentIndex == numAgents - 1:
		depth += -1
    nextAgentIndex = (agentIndex + 1)
    if nextAgentIndex == numAgents:
		nextAgentIndex = 0
    for action in gameState.getLegalActions(agentIndex):
		toNext = self.miniMax(gameState.generateSuccessor(agentIndex, action),
       depth, nextAgentIndex)
		actions.append((toNext, action))
    if agentIndex == 0: 
      return max(actions)
    else:                 
      return min(actions)  

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
 
    return self.miniMax(gameState, self.depth,0)[1]
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
        Your minimax agent with alpha-beta pruning (question 3)
        "*** YOUR CODE HERE ***"
    """
    def abmax(self, gameState , agentIndex , depth , alpha , beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        best = float("-infinity")
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            abNext = self.abmin(successor, agentIndex+1, depth, alpha, beta)
            best = max(best, abNext)
            if best > beta:
                return best
            alpha = max(best, alpha)
        return best
    
    def abmin(self, gameState , agentIndex, depth, alpha , beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        best = float("infinity")      
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                successor = gameState.generateSuccessor(agentIndex, action)
                abNext =self.abmax(successor, 0, depth+1, alpha, beta)
                best = min(best, abNext)
                if best < alpha:
                    return best
                beta = min(best,beta)
            else:
                successor = gameState.generateSuccessor(agentIndex, action)
                abNext = self.abmin(successor, agentIndex+1,depth,alpha, beta)
                best = min(best, abNext)
                if best < alpha:
                    return best
                beta = min(best,beta)
        return best
        
    def getAction(self,gameState):
        alpha = float("-infinity")
        beta = float("infinity")
        best = float("-infinity")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            alpbet = self.abmin(successor, self.index+1, 0, alpha, beta)
            if alpbet > best:
                best = alpbet
                path = action
            alpha = max(alpbet,alpha)
        return path

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def expectimax(self, gameState, depth, agentIndex):
    if gameState.isWin() or gameState.isLose() or depth == 0:
      return ( self.evaluationFunction(gameState), )
    numAgents = gameState.getNumAgents()
    if agentIndex == numAgents - 1:
		depth += -1
    nextAgentIndex = (agentIndex + 1)
    if nextAgentIndex == numAgents:
		nextAgentIndex = 0
    legalActions = gameState.getLegalActions(agentIndex)
    actions = [(self.expectimax(gameState.generateSuccessor(agentIndex, a), depth,
     nextAgentIndex)[0], a) for a in gameState.getLegalActions(agentIndex)]

    if(agentIndex == 0):
      return max(actions)
    else: 
      return (reduce(lambda x, y: x + y[0], actions, 0)/len(legalActions),)

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    return self.expectimax(gameState, self.depth, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    foodScore= 0
    ghostScore = 0
    foodist = 0
    dist = 0
    foods = newFood.asList()
    numFood = len(foods)
    for food in foods:
        foodist += manhattanDistance(food, newPos)
    score = currentGameState.getScore()
    if numFood == 0:
        score = 999999
    else:
		foodScore = 1/(1+numFood+foodist)
    if newScaredTimes[0] > 0:
        ghostScore += 999
    for ghost in newGhostStates:
        dist += manhattanDistance(newPos, ghost.getPosition())
        if ghost.scaredTimer == 0 and dist < 10:
            ghostScore -= 1/(10 - dist)
        else:
            ghostScore += 1/(dist)
    score += foodScore + ghostScore
    return score;

# Abbreviation
better = betterEvaluationFunction

