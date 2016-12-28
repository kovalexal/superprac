// Basic C++ headers
#include <iostream>
#include <fstream>
#include <numeric>
// Boost
#include <boost/mpi.hpp>
#include <boost/tokenizer.hpp>
#include <boost/serialization/set.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace boost::numeric::ublas;

// Used to calculate argmin and argmax over 1d values
typedef struct arg1d {
    friend class boost::serialization::access;
    int id;
    double value;
    arg1d(const int &id_in=0, const double &value_in=0) : id(id_in), value(value_in) {};

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & id; ar & value;
    }

    bool
    operator<(const arg1d &other) const {
        return value < other.value;
    }
} arg1d;

// Used to calculate argmin and argmax over 2d values
typedef struct arg2d {
    friend class boost::serialization::access;
    int id1, id2;
    double value;
    arg2d(const int &id1_in=0, const int &id2_in = 0, const double &value_in=0) : id1(id1_in), id2(id2_in), value(value_in) {};

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & id1; ar & id2; ar & value;
    }

    bool
    operator<(const arg2d &other) const {
        return value < other.value;
    }
} arg2d;

// Used to exit from a program by a true condition
void
check(bool condition, const std::string &message, const char &return_value) {
    if (condition) {
        std::cout << message << std::endl;
        exit(return_value);
    }
}

// Used to calculate cost of current configuration
double
calculate_cost(const matrix<double> &d, const std::set<size_t> &m) {
    // Get nonmedoids indexes
    std::set<size_t> nm;
    std::set_difference((boost::counting_iterator<size_t>(0)), (boost::counting_iterator<size_t>(d.size1())), m.begin(), m.end(), std::inserter(nm, nm.end()));

    // Calculate cost
    double cost = 0.0;
    std::for_each(nm.begin(), nm.end(), [&cost, &d, &m](const size_t &i) {
        size_t min_id = *std::min_element(m.begin(), m.end(), [&d, &i](const size_t &l1, const size_t &l2) {
            return d(i, l1) < d(i, l2);
        });
        cost += d(i, min_id);
    });

    return cost;
}

// Used to read distances matrix from csv file
template <typename T> matrix<T>
read_distances_csv(const std::string &infile) {
    // Open file for read
    std::ifstream in(infile.c_str(), std::ifstream::in);
    check((!in), "Problem during file opening", 1);

    // Used to store read values
    std::vector<T> values;
    size_t n_rows = 0;

    // Define a type to tokenize string
    typedef boost::tokenizer<boost::escaped_list_separator<char>> tokenizer;

    // Read file line by line
    std::string line;
    while (std::getline(in, line)) {
        tokenizer tok{line};
        for (const auto &t : tok)
            values.push_back(boost::lexical_cast<T>(t));
        n_rows++;
    }

    // Check that file contains matrix
    check((n_rows*n_rows != values.size()), "Wrong distances file", 1);

    // Create a matrix from values and return it
    unbounded_array<T> storage(n_rows*n_rows);
    std::copy(values.begin(), values.end(), storage.begin());
    return matrix<T>(n_rows, n_rows, storage);
}

// Used to break some domain over processes
std::tuple<size_t, size_t>
decompose_domain(int domain_size, int world_size, int world_rank) {
    //assert(domain_size >= world_size);

    size_t subdomain_start = 0, subdomain_size = domain_size / world_size;
    for (int i = 0; i < world_rank; ++i) {
        subdomain_start += subdomain_size;
        subdomain_size = (domain_size - subdomain_start) / (world_size - i - 1);
    }

    return std::make_tuple(subdomain_start, subdomain_size);
}

// Used to get position on 2d grid by absolute 1d position
std::tuple<size_t, size_t>
get_position(int k, int n_cols) {
    size_t i = k / n_cols, j = k - i * n_cols;
    return std::make_tuple(i, j);
};


int main(int argc, char *argv[])
{
    // Initialize boost::mpi environment
    boost::mpi::environment env{argc, argv};
    boost::mpi::communicator world;

    boost::mpi::timer global_timer;
    double read_time=0.0, build_time=0.0, swap_time=0.0;

    // Read csv file
    matrix<double> distances;
    size_t k = boost::lexical_cast<size_t>(argv[2]);
    if (world.rank() == 0) {
        distances = read_distances_csv<double>(argv[1]);
        read_time = global_timer.elapsed();
        std::cout.flush();
    }
    broadcast(world, distances, 0);
    global_timer.restart();
    int n_elems = distances.size1();

    // Used to measure execution time
    boost::mpi::timer algorithm_timer;

    /////////////////////////////
    //////// BUILD PHASE ////////
    /////////////////////////////
    std::set<size_t> medoids;
    std::vector<size_t> D(n_elems);

    // Parameters for current process
    size_t subdomain_start, subdomain_size, subdomain_end;
    std::tie(subdomain_start, subdomain_size) = decompose_domain(n_elems, world.size(), world.rank());
    subdomain_end = subdomain_start + subdomain_size;

    // Used in 0 process
    std::vector<arg1d> processes_candidates_1d(world.size());

    /////////////////////////////
    //////////// 1.1 ////////////
    /////////////////////////////
    // Find initial medoid

    // Calculate sum over rows for current process
    std::vector<arg1d> subdomain_values(subdomain_size);
    std::transform(boost::counting_iterator<size_t>(subdomain_start), boost::counting_iterator<size_t>(subdomain_end), subdomain_values.begin(), [&distances](const size_t &i) { return arg1d(i, sum(row(distances, i))); });

    // Find minimum for current process
    arg1d min_m = *std::min_element(subdomain_values.begin(), subdomain_values.end());

    // Find minumum over all processes
    gather(world, min_m, processes_candidates_1d, 0);
    if (world.rank() == 0) {
        arg1d first_medoid = *std::min_element(processes_candidates_1d.begin(), processes_candidates_1d.end());
        medoids.insert(first_medoid.id);
        std::fill(D.begin(), D.end(), first_medoid.id);
    }
    broadcast(world, medoids, 0);
    broadcast(world, D, 0);

    /////////////////////////////
    //////////// 1.2 ////////////
    /////////////////////////////
    // Find remaining medoids

    while (medoids.size() != k) {
        // Construct candidates for each process
        std::set<size_t> o_p, o((boost::counting_iterator<size_t>(subdomain_start)), (boost::counting_iterator<size_t>(subdomain_end)));
        std::set_difference(o.begin(), o.end(), medoids.begin(), medoids.end(), std::inserter(o_p, o_p.end()));

        // Calculate function for each candidate in process
        subdomain_values.resize(o_p.size());
        std::transform(o_p.begin(), o_p.end(), subdomain_values.begin(), [&distances, &D, &o_p](const size_t &i) {
            std::set<size_t> sum_domain(o_p);
            sum_domain.erase(i);

            double summ = 0.0;
            for (const auto &j: sum_domain) {
                summ += std::max(distances(j, D[j]) - distances(j, i), 0.0);
            }

            return arg1d(i, summ);
        });

        // Find maximum for current process
        arg1d max_m = *std::max_element(subdomain_values.begin(), subdomain_values.end());

        // Find maximum over all processes
        gather(world, max_m, processes_candidates_1d, 0);
        if (world.rank() == 0) {
            arg1d found_medoid = *std::max_element(processes_candidates_1d.begin(), processes_candidates_1d.end());
            medoids.insert(found_medoid.id);

            // Update information about closest medoid
            for (int i = 0; i < D.size(); ++i)
                if (distances(i, found_medoid.id) < distances(i, D[i]))
                    D[i] = found_medoid.id;
        }
        broadcast(world, medoids, 0);
        broadcast(world, D, 0);
    }

    //
    if (world.rank() == 0)
        build_time = global_timer.elapsed();
    global_timer.restart();

    /////////////////////////////
    //////// SWAP PHASE /////////
    /////////////////////////////

    size_t N_ITERATIONS = 0;

    // Used to stop iteration
    size_t remaining_iterations = boost::lexical_cast<size_t>(argv[3]);

    // Best cost is correct only on process 0
    double best_cost = DBL_MAX;
    if (world.rank() == 0)
        best_cost = calculate_cost(distances, medoids);

    // Used in process 0 to find best replacement candidate
    std::vector<arg2d> processes_candidates_2d(world.size());

    // Main iteration
    while (remaining_iterations > 0) {
        // Get current process calculation domain
        std::set<size_t> nonmedoids;
        std::set_difference(boost::counting_iterator<size_t>(0), boost::counting_iterator<size_t>(n_elems), medoids.begin(), medoids.end(), std::inserter(nonmedoids, nonmedoids.end()));
        std::tie(subdomain_start, subdomain_size) = decompose_domain(medoids.size() * nonmedoids.size(), world.size(), world.rank());
        subdomain_end = subdomain_start + subdomain_size;

        // Find best replacement candidate for each process
        arg2d replacement_candidate = arg2d(0, 0, DBL_MAX);
        for (size_t n = subdomain_start; n < subdomain_end; ++n) {
            // Find current i, h
            size_t i, h;
            std::tie(i, h) = get_position(n, nonmedoids.size());
            i = *std::next(medoids.begin(), i);
            h = *std::next(nonmedoids.begin(), h);

            // Calculate current configuration cost
            std::set<size_t> medoids_local(medoids);
            medoids_local.erase(i);
            medoids_local.insert(h);
            double cost_local = calculate_cost(distances, medoids_local);

            // Replace min
            if (cost_local < replacement_candidate.value)
                replacement_candidate = arg2d(i, h, cost_local);
        }
        gather(world, replacement_candidate, processes_candidates_2d, 0);

        // Process 0 makes decision
        if (world.rank() == 0) {
            arg2d best_candidate = *std::min_element(processes_candidates_2d.begin(), processes_candidates_2d.end());
            if (best_candidate.value < best_cost) {
                medoids.erase(best_candidate.id1); medoids.insert(best_candidate.id2);
                best_cost = best_candidate.value;
                remaining_iterations--;
            } else {
                remaining_iterations = 0;
            }
        }
        broadcast(world, medoids, 0);
        broadcast(world, remaining_iterations, 0);
        N_ITERATIONS++;
    }

    // Output result
    if (world.rank() == 0) {
        swap_time = global_timer.elapsed();

        double execution_time = algorithm_timer.elapsed();

         // Open file for write
        std::string outfile = argv[4];
        std::ofstream out(outfile.c_str(), std::ofstream::out);
        check((!out), "Problem during file opening", 1);

        out << "{\n";
        out << "\t\"infile\" :\" " << argv[1] << "\",\n";
        out << "\t\"n_elems\" : " << n_elems << ",\n";
        out << "\t\"n_proc\" : " << world.size() << ",\n";
        out << "\t\"k\" : " << k << ",\n";
        out << "\t\"n_iter\" : " << N_ITERATIONS << ",\n";
        out << "\t\"time\" : " << execution_time << ",\n";
        out << "\t\"times\" : [" << read_time << ", " << build_time << ", " << swap_time << "],\n";
        out << "\t\"result\" : [";
        for(auto it = medoids.begin(); it != medoids.end(); it=std::next(it, 1)){
            out << *it;
            if(std::next(it, 1) != medoids.end()){
                out << ", ";
            }
        }
        out << "]\n";
        out << "}\n";
        out.close();
    }
}
