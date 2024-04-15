import numpy as np

class ActorCritic:
    
    def __init__(self, env, order, max_episodes, alpha_w, alpha_t, lamb_w, lamb_t, gamma):
        self.env = env
        self.order = order
        self.max_episodes = max_episodes
        self.alpha_w = alpha_w
        self.alpha_t = alpha_t
        self.lamb_w = lamb_w
        self.lamb_t = lamb_t
        self.gamma = gamma
        
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.fourier_features = np.power(order + 1, self.num_states) # Number of fourier features: for n = 2 (81), for n = 1 (16)
        self.num_features = self.fourier_features * env.action_space.n # Total number of features for all actions
        
        # Normalize the state space
        self.min_vals = self.env.observation_space.low
        self.min_vals[1], self.min_vals[3] = -3.0, -4.0 # To make the velocity range between -3 and -4
        self.max_vals = self.env.observation_space.high
        self.max_vals[1], self.max_vals[3] = 3.0, 4.0 # To make the velocity range between 3 and 4

    def normalize(self, state):
        """
        Function to normalize the state space between 0 and 1
        """
        return (state - self.min_vals) / (self.max_vals - self.min_vals)

    def fourier_basis(self, state_vals):
        """
        Return X of size (n+1)^k * num_actions
        """
        index = 0
        X = np.zeros(self.fourier_features)
        for i in range(self.order + 1):
            for j in range(self.order + 1):
                for k in range(self.order + 1):
                    for l in range(self.order + 1):
                        X[index] = np.cos(np.pi * np.dot(state_vals, np.array([i, j, k, l])))
                        index += 1
        return np.tile(X, self.num_actions)

    def v_function(self, W, X):
        """
        Return Q(s, a) = W.T * X(s, a) which is a real number.
        """
        return np.dot(W, X)

    def get_start_end_indices(self, action):
        """
        Return start and end indices of W depending on the action taken.
        """
        start_idx = action * self.fourier_features
        end_idx = (action + 1) * self.fourier_features
        return start_idx, end_idx

    def get_one_hot_vector(self, action, x):
        """
        This function results in one-hot encoded vector of fourier features depending on the action taken. Return same size as X.
        """
        one_hot_vector = np.zeros(self.num_features)
        x_s = x
        s, e = self.get_start_end_indices(action)
        one_hot_vector[s:e] = x_s[s:e]
        return one_hot_vector

    def softmax(self, preferences):
        """
        Return softmax of preferences. Returns a vector of size num_actions with values that sum to 1.
        """
        exp_preferences = np.exp(preferences - np.max(preferences)) # To avoid overflow
        return exp_preferences / np.sum(exp_preferences)

    def select_action_gradient(self, theta, fourier_basis_features):
        """
        Return action (integer) based on the softmax of preferences and gradient of ln(pi(a|s)) wrt theta (same size as X).
        """
        all_preferences = np.zeros(self.num_actions)
        x = fourier_basis_features
        for action in range(self.num_actions):
            s, e = self.get_start_end_indices(action)
            all_preferences[action] = np.dot(theta[s:e], x[s:e]) # Preference = theta.T * X(s, a)
        action_probs = self.softmax(all_preferences)
        final_action = np.random.choice(self.num_actions, p = action_probs)
        
        # Compute the gradient (same size as X) (gradient = x_a - sum_over_actions(action_probs * x))
        x_a = self.get_one_hot_vector(final_action, x)
        summation = action_probs[0] * self.get_one_hot_vector(0, x) + action_probs[1] * self.get_one_hot_vector(1, x)
        gradient = x_a - summation
        return final_action, gradient
    
    def actor_critic_with_eligibility_traces(self):
        """
        Return the weights of the actor-critic, rewards and steps
        """
        omega = np.zeros(self.num_features)
        theta = np.zeros(self.num_features)
        # Initialize the reward and steps list
        reward_list = []
        step_list = []
        for eps in range(self.max_episodes):
            obs, _ = self.env.reset()
            obs = self.normalize(obs)
            r_sum = 0
            s_sum = 0
            # Initialize the eligibility traces
            z_omega = np.zeros(self.num_features)
            z_theta = np.zeros(self.num_features)
            cap_i = 1
            terminal = False
            while not terminal:
                X_s_omega = self.fourier_basis(obs)
                action, gradient = self.select_action_gradient(theta, X_s_omega)
                next_obs, reward, terminal, truncate, _ = self.env.step(action)
                next_obs = self.normalize(next_obs)
                X_sprime_omega = self.fourier_basis(next_obs)
                r_sum += reward
                s_sum += 1
                s, e = self.get_start_end_indices(action)
                delta = reward + self.gamma * self.v_function(omega[s:e], X_sprime_omega[s:e]) - self.v_function(omega[s:e], X_s_omega[s:e])
                z_omega[s:e] = self.gamma * self.lamb_w * z_omega[s:e] + X_s_omega[s:e]
                z_theta[s:e] = self.gamma * self.lamb_t * z_theta[s:e] + cap_i * gradient[s:e]
                # Update the weights
                omega[s:e] = omega[s:e] + self.alpha_w * delta * z_omega[s:e]
                theta[s:e] = theta[s:e] + self.alpha_t * delta * z_theta[s:e]
                # Update the variables
                cap_i = self.gamma * cap_i
                obs = next_obs
                if truncate:
                    break
            reward_list.append(r_sum)
            step_list.append(s_sum)
            
        return omega, theta, np.array(reward_list), np.array(step_list)