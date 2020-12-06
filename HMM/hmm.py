from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S, L, O = len(self.pi), len(Osequence), self.find_item(Osequence)
        alpha = np.zeros([S, L])

        # TODO: compute and return the forward messages alpha
        # Recursively resolve the forward message
        alpha[:, 0] = np.multiply(self.pi, self.B[:, O[0]])
        for t in range(1, L):
            for s in range(S):
                alpha[s, t] = np.multiply(self.B[s, O[t]], np.dot(self.A[:, s], alpha[:, t - 1]))

        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S, L, O = len(self.pi), len(Osequence), self.find_item(Osequence)
        beta = np.zeros([S, L])

        # TODO: compute and return the backward messages beta
        beta[:, L - 1] = np.ones(S)
        for t in range(L - 2, -1, -1):
            for s in range(S):
                beta[s, t] = np.sum(np.multiply(self.A[s, :], np.multiply(self.B[:, O[t + 1]], beta[:, t + 1])))

        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*T) A numpy array of observation sequence with length T

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """

        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        T = len(Osequence)

        alpha, beta = self.forward(Osequence=Osequence), self.backward(Osequence=Osequence)

        # For any t, the following formula should be true
        t = np.random.randint(0, T)
        P = np.sum(np.multiply(alpha[:, t], beta[:, t]))

        return P

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*T) A numpy array of observation sequence with length T

        Returns:
        - gamma: (num_state*T) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        # TODO: compute and return gamma using the forward/backward messages
        S, T = len(self.pi), len(Osequence)
        alpha, beta = self.forward(Osequence=Osequence), self.backward(Osequence=Osequence)

        # For any t, the following formula should be true
        t = np.random.randint(0, T)
        P = np.sum(np.multiply(alpha[:, t], beta[:, t]))

        gamma = np.zeros((S, T))
        for t in range(T):
            for s in range(S):
                gamma[s, t] = alpha[s, t] * beta[s, t] / P

        return gamma

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*T) A numpy array of observation sequence with length T

        Returns:
        - prob: (num_state*num_state*(T-1)) A numpy array where prob[i, j, t-1] =
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S, T, O = len(self.pi), len(Osequence), self.find_item(Osequence)
        prob = np.zeros([S, S, T - 1])

        # TODO: compute and return prob using the forward/backward messages
        alpha, beta = self.forward(Osequence=Osequence), self.backward(Osequence=Osequence)

        # For any t, the following formula should be true
        t = np.random.randint(0, T)
        P = np.sum(np.multiply(alpha[:, t], beta[:, t]))

        for t in range(T - 1):
            for s1 in range(S):
                for s2 in range(S):
                    prob[s1, s2, t] = alpha[s1, t] * self.A[s1, s2] * self.B[s2, O[t + 1]] * beta[s2, t + 1] / P

        return prob

    def most_likely_path_prob(self, Osequence):
        """
        Compute the probability for most likely path

        :param Osequence: (1*T) A numpy array of observation sequence with length T
        :return: prob: (num_state*T) A numpy array where prob[i, t-1] =
                    max P(Z_t = s, Z_{1:t−1} = z_{1:t−1}, X_{1:t} = x_{1:t})
        """
        S, T, O = len(self.pi), len(Osequence), self.find_item(Osequence)

        sigma, delta = np.zeros([S, T]), np.zeros([S, T], dtype=np.int64)
        # Recursively resolve probability for most likely path
        sigma[:, 0] = np.multiply(self.pi, self.B[:, O[0]])
        for t in range(1, T):
            for s in range(S):
                # State transition
                prod = np.multiply(self.A[:, s], sigma[:, t - 1])

                sigma[s, t] = self.B[s, O[t]] * np.max(prod)
                delta[s, t] = np.argmax(prod)

        return sigma, delta

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []

        # TODO: implement the Viterbi algorithm and return the most likely state path
        T = len(Osequence)
        sigma, delta = self.most_likely_path_prob(Osequence=Osequence)

        z_t = np.argmax(sigma[:, T - 1])
        path.append(self.find_key(self.obs_dict, z_t))
        for t in range(T - 2, -1, -1):
            z_t = delta[z_t, t + 1]
            path.append(self.find_key(self.obs_dict, z_t))

        path = path[::-1]
        return path

    # DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
