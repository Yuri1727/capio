#include "common.hpp"

TEST(multinodeIntegration, MapReduce) {
    const char *directory = getenv("CAPIO_DIR");
    std::vector<char> data;
    char path[4096];
    for (int i = 0; i < 10; ++i) {
        snprintf(path, sizeof(path), "%s/infile_%05d.dat", directory, i);
        FILE *file = fopen(path, "r");
        ASSERT_NE(file, nullptr);
        auto chunk = read_data(file);
        fclose(file);
        data.insert(data.end(), chunk.begin(), chunk.end());
    }
    ASSERT_FALSE(data.empty());
    ASSERT_EQ(write_data(data, 0.3, directory, 0, 2), 0);
}

int main(int argc, char **argv) { return run_tests(argc, argv); }
