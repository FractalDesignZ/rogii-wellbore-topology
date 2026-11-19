// Independent C++17 verifier for the public offset-path objective on small, exact-cost lattices.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using Path = std::vector<int>;
constexpr double inf = std::numeric_limits<double>::infinity();

struct Lattice {
    int rows, bins;
    std::vector<double> stations, transitions;

    double crossing(int row, int source, int dest) const {
        if (transitions.empty()) return 0;
        return transitions[(row * 3 + dest - source + 1) * bins + dest];
    }
};

struct Result {
    double cost = inf;
    Path path;
};

Result shortest_path(const Lattice& grid) {
    if (grid.rows == 0) return {0, {}};
    std::vector<double> prices(grid.stations.begin(), grid.stations.begin() + grid.bins);
    std::vector<int> parents(grid.stations.size(), -1);
    for (int t = 1; t < grid.rows; ++t) {
        std::vector<double> next(grid.bins, inf);
        for (int dest = 0; dest < grid.bins; ++dest) {
            int index = t * grid.bins + dest;
            for (int src = std::max(0, dest-1); src <= std::min(grid.bins-1, dest+1); ++src) {
                double price = prices[src] + grid.crossing(t-1, src, dest) + grid.stations[index];
                if (price < next[dest]) {
                    next[dest] = price;
                    parents[index] = src;
                }
            }
        }
        prices.swap(next);
    }
    auto end = std::min_element(prices.begin(), prices.end());
    if (!std::isfinite(*end)) return {};
    Path path(grid.rows);
    path.back() = static_cast<int>(end - prices.begin());
    for (int t = grid.rows-1; t > 0; --t) {
        path[t-1] = parents[t * grid.bins + path[t]];
    }
    return {*end, path};
}

// Visit every in-bounds path, including ones with disabled costs. Resolve equal
// total costs by lexicographic comparison from the terminal station backward.
void enumerate(const Lattice& grid, Path& path, Result& best, size_t& count) {
    const int t = static_cast<int>(path.size());
    if (t == grid.rows) {
        ++count;
        double cost = 0;
        for (int row = 0; row < t; ++row) {
            cost += grid.stations[row * grid.bins + path[row]];
            if (row > 0 && !grid.transitions.empty()) {
                const int move = path[row] - path[row-1];
                cost += grid.transitions[(row-1) * 3 * grid.bins + (1+move) * grid.bins + path[row]];
            }
        }
        if (std::isfinite(cost) && (cost < best.cost ||
            (cost == best.cost && std::lexicographical_compare(path.rbegin(), path.rend(),
                                                               best.path.rbegin(), best.path.rend())))) {
            best = {cost, path};
        }
        return;
    }
    for (int j = 0; j < grid.bins; ++j) {
        if (t && std::abs(j-path.back()) > 1) continue;
        path.push_back(j);
        enumerate(grid, path, best, count);
        path.pop_back();
    }
}

double read_cost(std::istream& in) {
    std::string token;
    if (!(in >> token)) throw std::runtime_error("missing cost");
    if (token == "x") return inf;
    size_t length = 0;
    double price = std::stod(token, &length);
    if (length != token.size() || !std::isfinite(price)) throw std::runtime_error("invalid cost");
    return price;
}

void check_case(std::istream& in, size_t& exhaustive_count) {
    Lattice grid{};
    if (!(in >> grid.rows >> grid.bins) || grid.rows < 0 || grid.rows > 8 ||
        grid.bins < 1 || grid.bins > 8) throw std::runtime_error("invalid fixture dimensions");
    for (int i = 0; i < grid.rows * grid.bins; ++i) grid.stations.push_back(read_cost(in));
    std::string marker;
    in >> marker;
    if (marker == "edges") {
        for (int i = 0; i < (grid.rows-1) * 3 * grid.bins; ++i) grid.transitions.push_back(read_cost(in));
    } else if (marker != "free") throw std::runtime_error("missing transition declaration");
    Result expected;
    expected.cost = read_cost(in);
    if (std::isfinite(expected.cost)) {
        for (int i = 0; i < grid.rows; ++i) {
            int bin = 0;
            if (!(in >> bin) || bin < 0 || bin >= grid.bins) throw std::runtime_error("invalid expected path");
            expected.path.push_back(bin);
        }
    }
    if (in >> marker) throw std::runtime_error("extra fixture fields");
    Path candidate;
    Result oracle;
    enumerate(grid, candidate, oracle, exhaustive_count);
    Result dp = shortest_path(grid);
    if (dp.cost != oracle.cost || dp.path != oracle.path || dp.cost != expected.cost ||
        dp.path != expected.path) throw std::runtime_error("Python, C++ and exhaustive optimum disagree");
}

int main(int argc, char** argv) {
    std::string current_case;
    try {
        if (argc != 2) throw std::runtime_error("expected the shared vector file");
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("cannot open vector file");
        std::unordered_set<std::string> names;
        size_t complete_paths = 0;
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream record(line);
            record >> current_case;
            if (!names.insert(current_case).second) throw std::runtime_error("duplicate case");
            check_case(record, complete_paths);
        }
        if (!input.eof() || names.size() != 1309) throw std::runtime_error("incomplete fixture corpus");
        std::cout << names.size() << " lattices, " << complete_paths
                  << " complete paths, Python path and cost parity passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << current_case << ": " << error.what() << '\n';
        return 1;
    }
}
