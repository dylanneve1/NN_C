#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x) {
    return x * (1.0f - x);
}

float random_normal(float mean, float stddev) {
    static float n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached) {
        float u1, u2;
        do {
            u1 = (float)rand() / RAND_MAX;
            u2 = (float)rand() / RAND_MAX;
        } while (u1 <= __FLT_EPSILON__);
        float mag = stddev * sqrtf(-2.0 * logf(u1));
        n2 = mag * sinf(2.0 * M_PI * u2);
        n2_cached = 1;
        return mean + mag * cosf(2.0 * M_PI * u2);
    } else {
        n2_cached = 0;
        return mean + n2;
    }
}

int main() {
    srand(time(NULL));

    int input_neurons = 26;
    int hidden_neurons = 50;
    int output_neurons = 1;
    float learning_rate = 0.005f;

    // Expanded training data (add more words as needed)
    char *words[] = {"hello", "world", "neural", "network", "programming", "test", "example", "learn", "code", "computer", "science", "algorithm", "data", "structure", "artificial", "intelligence", "machine", "learning", "deep", "python", "java", "javascript", "html", "css", "php", "ruby", "swift", "kotlin", "go", "rust", "typescript", "react", "angular", "vue", "node", "express", "mongodb", "sql", "postgresql", "mysql", "amazon", "google", "microsoft", "apple", "facebook", "twitter", "instagram", "linkedin", "github", "youtube", "wikipedia", "mathematics", "physics", "chemistry", "biology", "history", "geography", "economics", "politics", "philosophy", "literature", "art", "music", "sports", "food", "travel", "nature", "space", "universe", "innovation", "technology", "future", "education", "knowledge", "communication", "community", "collaboration", "open", "source", "internet", "security", "cloud", "computing", "big", "data", "analytics", "cybersecurity", "quantum", "blockchain", "cryptocurrency", "virtual", "reality", "augmented", "reality", "robotics", "automation", "biotechnology", "nanotechnology", "genetic", "engineering", "renewable", "energy", "sustainable", "development", "global", "warming", "climate", "change", "environment", "ecology", "medical", "health", "research", "innovation", "science", "technology", "engineering", "mathematics"}; // More words added

    int num_words = sizeof(words) / sizeof(words[0]);

    float hidden_weights[50][26];
    float hidden_biases[50];
    float output_weights[50];
    float output_bias;

    // Initialize weights and biases
    for (int i = 0; i < hidden_neurons; i++) {
        for (int j = 0; j < input_neurons; j++) {
            hidden_weights[i][j] = random_normal(0.0f, 1.0f / sqrtf(input_neurons));
        }
        hidden_biases[i] = 0.0f;
    }
    for (int j = 0; j < hidden_neurons; j++) {
        output_weights[j] = random_normal(0.0f, 1.0f / sqrtf(hidden_neurons));
    }
    output_bias = 0.0f;

    int epochs = 10000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < num_words; i++) {
            float inputs[26] = {0.0f};
            for (int j = 0; j < strlen(words[i]); j++) {
                char c = tolower(words[i][j]);
                if (isalpha(c)) {
                    inputs[c - 'a'] += 1.0f; // count of char
                }
            }

            // Forward pass
            float hidden_outputs[50];
            for (int j = 0; j < hidden_neurons; j++) {
                float sum = hidden_biases[j];
                for (int k = 0; k < input_neurons; k++) {
                    sum += inputs[k] * hidden_weights[j][k];
                }
                hidden_outputs[j] = sigmoid(sum);
            }

            float output_sum = output_bias;
            for (int j = 0; j < hidden_neurons; j++) {
                output_sum += hidden_outputs[j] * output_weights[j];
            }
            float output = sigmoid(output_sum);


            float target = (float)strlen(words[i]) / 26.0f;


            // Backpropagation
            float output_error = target - output;
            float output_delta = output_error * sigmoid_derivative(output);

            float hidden_deltas[50];
            for (int j = 0; j < hidden_neurons; j++) {
                float error = output_delta * output_weights[j];
                hidden_deltas[j] = error * sigmoid_derivative(hidden_outputs[j]);
            }

            // Update weights and biases
            for (int j = 0; j < hidden_neurons; j++) {
                for (int k = 0; k < input_neurons; k++) {
                    hidden_weights[j][k] += learning_rate * hidden_deltas[j] * inputs[k];
                }
                hidden_biases[j] += learning_rate * hidden_deltas[j];
            }

            for (int j = 0; j < hidden_neurons; j++) {
                output_weights[j] += learning_rate * output_delta * hidden_outputs[j];
            }
            output_bias += learning_rate * output_delta;
        }
    }

    // Testing
    char test_word[50];
    printf("Enter a word: ");
    scanf("%s", test_word);

    float test_inputs[26] = {0.0f};
    for (int j = 0; j < strlen(test_word); j++) {
        char c = tolower(test_word[j]);
        if (isalpha(c)) {
             test_inputs[c - 'a'] += 1.0f;
        }
    }

    float test_hidden_outputs[50];
    for (int j = 0; j < hidden_neurons; j++) {
        float sum = hidden_biases[j];
        for (int k = 0; k < input_neurons; k++) {
            sum += test_inputs[k] * hidden_weights[j][k];
        }
        test_hidden_outputs[j] = sigmoid(sum);
    }

    float test_output_sum = output_bias;
    for (int j = 0; j < hidden_neurons; j++) {
        test_output_sum += test_hidden_outputs[j] * output_weights[j];
    }
    float test_output = sigmoid(test_output_sum);

    printf("Predicted letter count: %f\n", test_output * 26);

    return 0;
}