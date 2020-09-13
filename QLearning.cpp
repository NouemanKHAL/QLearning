#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#pragma GCC target ("avx2")
#pragma GCC optimization ("O3")
#pragma GCC optimization ("unroll-loops")
#define debug(x) cout << #x << ":" << x << ' ';
#define debugg(x) cout << #x << ":" << x << ' ' << "\n";
#define endl '\n'

using namespace std;

const int INF = 0x3f3f3f3f; // around 10^9, used as INFINITY

// O, N, S, E
int next_state_id[9][4] = {
	{0, 0, 3, 1},
	{0, 1, 4, 2},
	{0, 2, 5, 2},
	{3, 0, 6, 4},
	{3, 1, 7, 5},
	{4, 2, 8, 5},
	{6, 3, 6, 7},
	{6, 4, 7, 8},
	{7, 5, 8, 8}
};

double reward[9][4] = {
	{1.0, 1.0, 1.0, -1.0},
	{1.0, -1.0, -1.0, 10.0},
	{-1.0, 10.0, 1.0, 10.},
	{1.0, 1.0, 1.0, -1.0},
	{1.0, -1.0, 1.0, 1.0},
	{-1.0, 10.0, 1.0, 1.0},
	{1.0, 1.0, 1.0, 1.0},
	{1.0, -1.0, 1.0, 1.0},
	{1.0, 1.0, 1.0, 1.0}
};

// helps printing the Q-table
void print2DVect(vector<vector<double>> & q) {
	for (auto &vect : q) {
		for (auto &x : vect) {
			cout << setw(8) << setprecision(5) << x << ' ';
		}
		cout << endl;
	}
}

pair<int, int> getRandomNextState(int state) {
	int action = rand() % 4;
	return {next_state_id[state][action], action};
}

pair<int, int> getMaximumNextState(int state, vector<vector<double>> & q) {

	double current_maximum = -INF;
	int action;

	for (int i = 0 ; i < 4 ; ++i) {
		current_maximum = max(current_maximum, q[state][i]);
	}

	for (int i = 0 ; i < 4 ; ++i) {
		if (q[state][i] == current_maximum) {
			action = i;
			break;
		}
	}

	return {next_state_id[state][action], action};
}

vector<vector<double>> qlearning(double learning_rate, double discount_factor, double epsilon, int episodes, double & score) {

	// initial Q-table of size(9, 4) filled with zeros
	vector<vector<double>> q(9, vector<double>(4, 0.0));


	while (episodes-- > 0) {

		int state = 0;

		score = 0.0;

		while (state != 2) {
			double r = ((double) rand() / (RAND_MAX));

			pair<int, int> tmp;

			// epsilon-greedy strategy
			if (r < 1 - epsilon) {
				tmp = getMaximumNextState(state, q);
			} else {
				tmp = getRandomNextState(state);
			}

			int next_state = tmp.first;
			int action = tmp.second;


			double maximum = -INF;
			for (int i = 0 ; i < 4 ; ++i) {
				maximum = max(maximum, q[next_state][i]);
			}


			// updating the Q-table
			q[state][action] += learning_rate * (reward[state][action] + discount_factor * maximum - q[state][action]);

			// updating the score
			score += reward[state][action];

			// updating the values for next iteration
			state = next_state;
		}

		epsilon *= 0.99;
		learning_rate *= 0.99;
	}

	return q;
}


int main() {
	// input/output optimization
	ios_base::sync_with_stdio(false);
	cin.tie(nullptr);


	// initializing the random generator with a random seed based on current time
	srand(time(NULL));


	// initializing the parameters of the Q-learning algorithm
	double learning_rate = 1.0, epsilon = 1.0, discount_factor = 0.2;


	double score = 0.0;

	// Q-learning algorithm -- 5 Episodes
	auto first_q = qlearning(learning_rate, discount_factor, epsilon, 5, score);
	print2DVect(first_q);

	cout << "---------------- Score : " << score << " ----------------" <<  endl;

	// Q-learning algorithm -- 50 Episodes
	auto second_q = qlearning(learning_rate, discount_factor, epsilon, 50, score);
	print2DVect(second_q);

	cout << "---------------- Score : " << score << " ----------------" << endl;

	/*
		1- in the first Q-table, we notice that the max values are both equal to 1.25 and are both leading to stay in the same cell ==> no strategy exists
		2- in the second Q-table, the first line's max value is 1.2528 corresponding to South, leading us to the next max value in the 4th line 1.264 corresponding to south, then the next max value is again 1.264 leading to south, 
		then in the 7th line we find the max value 1.32 leading to east etc...
		We can retrieve the best strategy following the Max values as the following path :
			a1 -> b1 -> c1 -> c2 -> c3 -> b3 -> a3

	*/

	return 0;
}