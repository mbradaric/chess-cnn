#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

#define MAX_LINE_LENGTH 4096
#define MAX_MOVES_LENGTH 65536
#define INITIAL_GAMES_CAPACITY 10000

/*
 * PGN to CSV Converter
 * ====================
 * Extracts moves from a PGN file and creates a CSV file with an AN column
 *
 * Usage: ./pgn_to_csv input.pgn output.csv
 *
 * Compile: gcc -o pgn_to_csv pgn_to_csv.c
 */

typedef struct {
    char *moves;
} Game;

typedef struct {
    Game *games;
    size_t count;
    size_t capacity;
} GameList;

// Initialize the game list
void init_game_list(GameList *list) {
    list->capacity = INITIAL_GAMES_CAPACITY;
    list->count = 0;
    list->games = malloc(list->capacity * sizeof(Game));
    if (!list->games) {
        fprintf(stderr, "Error: Failed to allocate memory for game list\n");
        exit(1);
    }
}

// Add a game to the list
void add_game(GameList *list, const char *moves) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->games = realloc(list->games, list->capacity * sizeof(Game));
        if (!list->games) {
            fprintf(stderr, "Error: Failed to reallocate memory for game list\n");
            exit(1);
        }
    }
    
    list->games[list->count].moves = strdup(moves);
    if (!list->games[list->count].moves) {
        fprintf(stderr, "Error: Failed to allocate memory for moves\n");
        exit(1);
    }
    list->count++;
}

// Free the game list
void free_game_list(GameList *list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->games[i].moves);
    }
    free(list->games);
    list->games = NULL;
    list->count = 0;
    list->capacity = 0;
}

// Check if a line is a PGN tag (starts with "[")
bool is_tag_line(const char *line) {
    // Skip leading whitespac
    while (*line && isspace(*line)) {
        line++;
    }
    return *line == '[';
}

// Check if a line is empty or contains only whitespace
bool is_empty_line(const char *line) {
    while (*line) {
        if (!isspace(*line)) {
            return false;
        }
        line++;
    }
    return true;
}

// Trim leading and trailing whitespace from a string
void trim(char *str) {
    char *start = str;
    char *end;
    
    // Trim leading whitespace
    while (*start && isspace(*start)) {
        start++;
    }
    
    // If string is all whitespace
    if (*start == '\0') {
        *str = '\0';
        return;
    }
    
    // Move the trimmed string to the beginning
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }
    
    // Trim trailing whitespace
    end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) {
        *end = '\0';
        end--;
    }
}

// Remove game result from the end of moves string (1-0, 0-1, 1/2-1/2, *) ((not necessary but won't hurt))
void remove_result(char *moves) {
    size_t len = strlen(moves);
    if (len == 0) return;
    
    // Find the last non-whitespace character
    char *end = moves + len - 1;
    while (end > moves && isspace(*end)) {
        end--;
    }
    
    // Check for various result patterns
    const char *results[] = {"1-0", "0-1", "1/2-1/2", "*"};
    int result_lens[] = {3, 3, 7, 1};
    
    for (int i = 0; i < 4; i++) {
        int rlen = result_lens[i];
        if (end - moves + 1 >= rlen) {
            char *check = end - rlen + 1;
            if (strncmp(check, results[i], rlen) == 0) {
                // Found result, remove it
                *check = '\0';
                trim(moves);
                return;
            }
        }
    }
}

// Escape double quotes for CSV output
void escape_csv_field(const char *input, char *output, size_t output_size) {
    size_t j = 0;
    output[j++] = '"';
    
    for (size_t i = 0; input[i] && j < output_size - 2; i++) {
        if (input[i] == '"') {
            // Escape double quote with another double quote
            if (j < output_size - 3) {
                output[j++] = '"';
                output[j++] = '"';
            }
        } else {
            output[j++] = input[i];
        }
    }
    
    output[j++] = '"';
    output[j] = '\0';
}

// Parse the PGN file and extract games
int parse_pgn_file(const char *filename, GameList *games) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return -1;
    }
    
    char line[MAX_LINE_LENGTH];
    char moves_buffer[MAX_MOVES_LENGTH];
    bool in_moves_section = false;
    bool found_any_tags = false;
    
    moves_buffer[0] = '\0';
    
    while (fgets(line, sizeof(line), fp)) {
        // Remove newline characters
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        
        if (is_tag_line(line)) {
            // If we were in moves section and have moves, save the game
            if (in_moves_section && strlen(moves_buffer) > 0) {
                trim(moves_buffer);
                remove_result(moves_buffer);
                if (strlen(moves_buffer) > 0) {
                    add_game(games, moves_buffer);
                }
                moves_buffer[0] = '\0';
            }
            in_moves_section = false;
            found_any_tags = true;
        } else if (is_empty_line(line)) {
            // Empty line after tags signals start of moves section
            if (found_any_tags && !in_moves_section) {
                in_moves_section = true;
            }
        } else if (in_moves_section) {
            // This is a moves line
            if (strlen(moves_buffer) > 0) {
                // Add space before appending
                size_t current_len = strlen(moves_buffer);
                if (current_len + 1 < MAX_MOVES_LENGTH) {
                    moves_buffer[current_len] = ' ';
                    moves_buffer[current_len + 1] = '\0';
                }
            }
            
            // Append the line to moves buffer
            size_t current_len = strlen(moves_buffer);
            size_t line_len = strlen(line);
            if (current_len + line_len < MAX_MOVES_LENGTH) {
                strcat(moves_buffer, line);
            }
        }
    }
    
    // Don't forget the last game
    if (strlen(moves_buffer) > 0) {
        trim(moves_buffer);
        remove_result(moves_buffer);
        if (strlen(moves_buffer) > 0) {
            add_game(games, moves_buffer);
        }
    }
    
    fclose(fp);
    return 0;
}

// Write games to CSV file
int write_csv_file(const char *filename, const GameList *games) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create file '%s'\n", filename);
        return -1;
    }
    
    // Write header
    fprintf(fp, "AN\n");
    
    // Write each game
    char *escaped = malloc(MAX_MOVES_LENGTH * 2);
    if (!escaped) {
        fprintf(stderr, "Error: Failed to allocate memory for CSV output\n");
        fclose(fp);
        return -1;
    }
    
    for (size_t i = 0; i < games->count; i++) {
        // Check if the moves contain comma, quote, or newline
        bool needs_quoting = false;
        const char *moves = games->games[i].moves;
        for (const char *p = moves; *p; p++) {
            if (*p == ',' || *p == '"' || *p == '\n' || *p == '\r') {
                needs_quoting = true;
                break;
            }
        }
        
        if (needs_quoting) {
            escape_csv_field(moves, escaped, MAX_MOVES_LENGTH * 2);
            fprintf(fp, "%s\n", escaped);
        } else {
            fprintf(fp, "%s\n", moves);
        }
    }
    
    free(escaped);
    fclose(fp);
    return 0;
}

void print_usage(const char *program_name) {
    printf("PGN to CSV Converter\n");
    printf("====================\n");
    printf("Extracts moves from a PGN file and creates a CSV file.\n\n");
    printf("Usage: %s <input.pgn> <output.csv>\n\n", program_name);
    printf("Arguments:\n");
    printf("  input.pgn   - Path to the input PGN file\n");
    printf("  output.csv  - Path to the output CSV file\n\n");
    printf("Example:\n");
    printf("  %s chess_games.pgn games.csv\n", program_name);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    
    printf("PGN to CSV Converter\n");
    printf("====================\n\n");
    
    // Initialize game list
    GameList games;
    init_game_list(&games);
    
    // Parse PGN file
    printf("Reading PGN file: %s\n", input_file);
    if (parse_pgn_file(input_file, &games) != 0) {
        free_game_list(&games);
        return 1;
    }
    printf("Found %zu games\n", games.count);
    
    // Write CSV file
    printf("Writing CSV file: %s\n", output_file);
    if (write_csv_file(output_file, &games) != 0) {
        free_game_list(&games);
        return 1;
    }
    
    printf("\nDone! Successfully converted %zu games.\n", games.count);
    
    // Cleanup
    free_game_list(&games);
    
    return 0;
}
