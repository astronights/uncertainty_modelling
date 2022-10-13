""" CS5340 Lab 1: Belief Propagation and Maximal Probability
See accompanying PDF for instructions.

Name: Shubhankar Agrawal
Email: e0925482@u.nus.edu
Student ID: A0248330L
"""

import copy
from typing import List

import numpy as np
import networkx as nx

from factor import Factor, index_to_assignment, assignment_to_index, generate_graph_from_factors, \
    visualize_graph


"""For sum product message passing"""
def factor_product(A, B):
    """Compute product of two factors.

    Suppose A = phi(X_1, X_2), B = phi(X_2, X_3), the function should return
    phi(X_1, X_2, X_3)
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor product
    Hint: The code for this function should be very short (~1 line). Try to
      understand what the above lines are doing, in order to implement
      subsequent parts.
    """
    out.val = A.val[idxA] * B.val[idxB]
    return out


def factor_marginalize(factor, var):
    """Sums over a list of variables.

    Args:
        factor (Factor): Input factor
        var (List): Variables to marginalize out

    Returns:
        out: Factor with variables in 'var' marginalized out.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var
    """

    # The out Factor variables, cardinality and values are initialized
    out.var = np.setdiff1d(factor.var, var)
    out.card = factor.card[np.isin(factor.var, out.var)]
    out.val = np.zeros(np.prod(out.card))
    assignments = factor.get_all_assignments()
    
    # For each assignment in the output, calculating the sum of rows from the input factor the assignment matches
    for i in out.get_all_assignments():
        vals = assignments[:, np.isin(factor.var, out.var)] == i
        out.val[assignment_to_index(i, out.card)] = np.sum(factor.val[vals.flatten()])
    return out


def observe_evidence(factors, evidence=None):
    """Modify a set of factors given some evidence

    Args:
        factors (List[Factor]): List of input factors
        evidence (Dict): Dictionary, where the keys are the observed variables
          and the values are the observed values.

    Returns:
        List of factors after observing evidence
    """
    if evidence is None:
        return factors
    out = copy.deepcopy(factors)

    """ YOUR CODE HERE
    Set the probabilities of assignments which are inconsistent with the 
    evidence to zero.
    """

    for factor in out:
        assignments = factor.get_all_assignments()
        for assignment in assignments:
            match_vals = []
            # For each assignment, checking if any of the evidence variables don't match the observed value
            for key in evidence.keys():
                match_vals.append(assignment[(factor.var == key)] != evidence[key])
            if any(match_vals):
                factor.val[assignment_to_index(assignment, factor.card)] = 0

    return out


"""For max sum meessage passing (for MAP)"""
def factor_sum(A, B):
    """Same as factor_product, but sums instead of multiplies
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor sum. The code for this
    should be very similar to the factor_product().
    """
    out.val = A.val[idxA] + B.val[idxB]

    # The val_argmax fields are merged between the variables in the two input factors

    # If the val_argmax is None, we initialize with a list of empty dictionaries
    A.val_argmax = [{}] * np.prod(A.card) if A.val_argmax is None else A.val_argmax
    B.val_argmax = [{}] * np.prod(B.card) if B.val_argmax is None else B.val_argmax

    # The val_argmax values are then merged between the two factors itemwise
    new_vals_argmax = []
    for i in range(len(out.val)):
        new_vals_argmax.append({**A.val_argmax[idxA[i]],**B.val_argmax[idxB[i]]})
    out.val_argmax = new_vals_argmax

    # If both do not have val_argmax, we set it to None
    if all([x == {} for x in out.val_argmax]):
        out.val_argmax = None

    return out


def factor_max_marginalize(factor, var):
    """Marginalize over a list of variables by taking the max.

    Args:
        factor (Factor): Input factor
        var (List): Variable to marginalize out.

    Returns:
        out: Factor with variables in 'var' marginalized out. The factor's
          .val_argmax field should be a list of dictionary that keep track
          of the maximizing values of the marginalized variables.
          e.g. when out.val_argmax[i][j] = k, this means that
            when assignments of out is index_to_assignment[i],
            variable j has a maximizing value of k.
          See test_lab1.py::test_factor_max_marginalize() for an example.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var. 
    You should make use of val_argmax to keep track of the location with the
    maximum probability.
    """

    # The out Factor variables, cardinality and values are initialized
    out.var = np.setdiff1d(factor.var, var)
    out.card = factor.card[np.isin(factor.var, out.var)]
    out.val = np.full(np.prod(out.card), -np.inf)
    out.val_argmax = []
    assignments = factor.get_all_assignments()

    for i in out.get_all_assignments():

        # The rows from the input factor the assignment matches are identified
        pos = np.argwhere((assignments[:, np.isin(factor.var, out.var)] == i).flatten() == True).flatten()
        vals = [(factor.val[x] if x in pos else -np.inf) for x in range(len(factor.val))]

        # The maximum value is identified and the corresponding assignment is stored in the val_argmax
        max_pos = np.argmax(vals).flatten()
        out.val[assignment_to_index(i, out.card)] = factor.val[max_pos]
        out.val_argmax.append(dict(zip([x for x in var if x in factor.var], index_to_assignment(max_pos, factor.card).flatten()[np.isin(factor.var, var)])))

        # If val_argmax previously existed, we update the dictionary with the new values
        if(factor.val_argmax is not None):
            out.val_argmax[-1].update(factor.val_argmax[max_pos[0]])

    return out


def compute_joint_distribution(factors):
    """Computes the joint distribution defined by a list of given factors

    Args:
        factors (List[Factor]): List of factors

    Returns:
        Factor containing the joint distribution of the input factor list
    """
    joint = Factor()

    """ YOUR CODE HERE
    Compute the joint distribution from the list of factors. You may assume
    that the input factors are valid so no input checking is required.
    """
    joint = factors[0]
    for factor in factors[1:]:
        joint = factor_product(joint, factor)

    return joint


def compute_marginals_naive(V, factors, evidence):
    """Computes the marginal over a set of given variables

    Args:
        V (int): Single Variable to perform inference on
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k] = v indicates that
          variable k has the value v.

    Returns:
        Factor representing the marginals
    """

    output = Factor()

    """ YOUR CODE HERE
    Compute the marginal. Output should be a factor.
    Remember to normalize the probabilities!
    """
    observed_factors = observe_evidence(factors, evidence)
    joint_observed_factors = compute_joint_distribution(observed_factors)
    output = factor_marginalize(joint_observed_factors, np.setdiff1d(joint_observed_factors.var, V))
    output.val = output.val/np.sum(output.val) # Normalization

    return output


def compute_marginals_bp(V, factors, evidence):
    """Compute single node marginals for multiple variables
    using sum-product belief propagation algorithm

    Args:
        V (List): Variables to infer single node marginals for
        factors (List[Factor]): List of factors representing the grpahical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        marginals: List of factors. The ordering of the factors should follow
          that of V, i.e. marginals[i] should be the factor for variable V[i].
    """
    # Dummy outputs, you should overwrite this with the correct factors
    marginals = []
    # Setting up messages which will be passed
    factors = observe_evidence(factors, evidence)
    graph = generate_graph_from_factors(factors)

    # Uncomment the following line to visualize the graph. Note that we create
    # an undirected graph regardless of the input graph since 1) this
    # facilitates graph traversal, and 2) the algorithm for undirected and
    # directed graphs is essentially the same for tree-like graphs.
    # visualize_graph(graph)

    # You can use any node as the root since the graph is a tree. For simplicity
    # we always use node 0 for this assignment.
    root = 0

    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    """ YOUR CODE HERE
    Use the algorithm from lecture 4 and perform message passing over the entire
    graph. Recall the message passing protocol, that a node can only send a
    message to a neighboring node only when it has received messages from all
    its other neighbors.
    Since the provided graphical model is a tree, we can use a two-phase 
    approach. First we send messages inward from leaves towards the root.
    After this is done, we can send messages from the root node outward.
    
    Hint: You might find it useful to add auxilliary functions. You may add 
      them as either inner (nested) or external functions.
    """

    # The pass_message function traverses through a defined order of nodes
    # For each node it picks the neighbors that are ahead of it in the traversal order and generates the message
    # The message is computed by collecting the messages from the children, the node -> neighbor and if the node contains a unary factor
    # The messages are joint and then marginalized over the children
    def pass_message(order_nodes):
        for node in order_nodes:
            for neighbor in graph.neighbors(node):
                if(order_nodes.index(node) < order_nodes.index(neighbor)):
                    children = [x for x in list(graph.neighbors(node)) if x != neighbor]
                    factors_to_multiply = [graph.edges[node, neighbor]['factor']] + [messages[child][node] for child in children]
                    if('factor' in graph.nodes[node]):
                        factors_to_multiply.append(graph.nodes[node]['factor'])
                    messages[node][neighbor] = factor_marginalize(compute_joint_distribution(factors_to_multiply), children+[node])
                    
    # DFS is used to calculate the post order (leaves -> root) and pre order (root -> leaves) traversal orders
    post_order_nodes = list(nx.dfs_postorder_nodes(graph, root))
    pass_message(post_order_nodes)
    
    pre_order_nodes = list(nx.dfs_preorder_nodes(graph, root))
    pass_message(pre_order_nodes)

    # The marginals are computed similar to the pass_message above and finally normalized
    for v in V:
        neighbors = list(graph.neighbors(v))
        messages_to_multiply = [messages[neighbor][v] for neighbor in neighbors]
        if('factor' in graph.nodes[v]):
            messages_to_multiply.append(graph.nodes[v]['factor'])
        joint_prob = compute_joint_distribution(messages_to_multiply)
        prob_factor = factor_marginalize(joint_prob, [x for x in joint_prob.var if x != v])
        prob_factor.val = prob_factor.val/np.sum(prob_factor.val)
        marginals.append(prob_factor)

    return marginals


def map_eliminate(factors, evidence):
    """Obtains the maximum a posteriori configuration for a tree graph
    given optional evidence

    Args:
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        max_decoding (Dict): MAP configuration
        log_prob_max: Log probability of MAP configuration. Note that this is
          log p(MAP, e) instead of p(MAP|e), i.e. it is the unnormalized
          representation of the conditional probability.
    """

    max_decoding = {}
    log_prob_max = 0.0

    """ YOUR CODE HERE
    Use the algorithm from lecture 5 and perform message passing over the entire
    graph to obtain the MAP configuration. Again, recall the message passing 
    protocol.
    Your code should be similar to compute_marginals_bp().
    To avoid underflow, first transform the factors in the probabilities
    to **log scale** and perform all operations on log scale instead.
    You may ignore the warning for taking log of zero, that is the desired
    behavior.
    """
    factors = observe_evidence(factors, evidence)
    graph = generate_graph_from_factors(factors)

    root = 0

    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    # Function to create new Factor with log of values
    def prob_to_log(factor):
        return Factor(factor.var, factor.card, np.log(factor.val), factor.val_argmax)

    # Function to calculate sum of a set of factors
    def sum_factors(factors):
        added = factors[0]
        for factor in factors[1:]:
            added = factor_sum(added, factor)
        return(added)

    # The pass_message function traverses through a defined order of nodes
    # For each node it picks the neighbors that are ahead of it in the traversal order and generates the message
    # The message is computed by collecting the messages from the children, the node -> neighbor and if the node contains a unary factor
    # The messages are summed and then max marginalized over the children
    def pass_message(order_nodes):
        for node in order_nodes:
            for neighbor in graph.neighbors(node):
                if(order_nodes.index(node) < order_nodes.index(neighbor)):
                    children = [x for x in list(graph.neighbors(node)) if x != neighbor]
                    factors_to_add = [prob_to_log(graph.edges[node, neighbor]['factor'])] + [messages[child][node] for child in children]
                    if('factor' in graph.nodes[node]):
                        factors_to_add.append(prob_to_log(graph.nodes[node]['factor']))
                    messages[node][neighbor] = factor_max_marginalize(sum_factors(factors_to_add), children+[node])
                    
    # Only the leaves -> root path is traversed using DFS
    post_order_nodes = list(nx.dfs_postorder_nodes(graph, root))
    pass_message(post_order_nodes)

    # The max marginal is computed similar to the pass_message above
    neighbors = list(graph.neighbors(root))
    messages_to_add = [messages[neighbor][root] for neighbor in neighbors]

    if('factor' in graph.nodes[root]):
        messages_to_add.append(prob_to_log(graph.nodes[root]['factor']))
    
    joint_prob = sum_factors(messages_to_add)
    prob_factor = factor_max_marginalize(joint_prob, [x for x in joint_prob.var if x != root])
    
    # The max marginal is then used to find the MAP configuration
    max_decoding = prob_factor.val_argmax[np.argmax(prob_factor.val)]
    max_decoding[root] = np.argmax(prob_factor.val)
    log_prob_max = np.max(prob_factor.val)

    for e in evidence.keys():
        del max_decoding[e]

    return max_decoding, log_prob_max
