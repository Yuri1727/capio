#include "remote/backend/mpi.hpp"
#include "utils/capiocl_adapter.hpp"
#include "utils/server_println.hpp"

namespace {
// Resolving an unknown rank through operator[] would hand back an empty hostname, which
// then resolves back to rank 0 when the reply is addressed - misrouting the reply instead
// of reporting the problem. Every lookup goes through here so that cannot happen silently.
const std::string &hostname_of_rank(const std::unordered_map<int, std::string> &rank_to_hostname,
                                    const int source_rank) {
    const auto entry = rank_to_hostname.find(source_rank);
    if (entry == rank_to_hostname.end()) {
        ERR_EXIT("Error: received a message from rank %d, whose hostname is not known",
                 source_rank);
    }
    return entry->second;
}
} // namespace

MPIBackend::MPIBackend(int argc, char **argv) : Backend(MPI_MAX_PROCESSOR_NAME) {
    int node_name_len, provided;
    START_LOG(gettid(), "call()");
    LOG("Created a MPI backend");
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &n_servers);
    LOG("Does MPI has multithreading support? %s (%d)",
        provided == MPI_THREAD_MULTIPLE ? "yes" : "no", provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    LOG("node_rank=%d", &rank);
    if (provided != MPI_THREAD_MULTIPLE) {
        LOG("Error: The threading support level is not MPI_THREAD_MULTIPLE (is %d)", provided);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Get_processor_name(node_name.data(), &node_name_len);
    LOG("Node name = %s, length=%d", node_name.data(), node_name_len);
    nodes.emplace(node_name);
    rank_to_hostname[rank]      = node_name;
    hostname_to_rank[node_name] = rank;
    server_println("initialization completed.", CapioCLEngine::get().getWorkflowName(),
                   CAPIO_LOG_SERVER_CLI_LEVEL_STATUS, "MPIBackend");
}

MPIBackend::~MPIBackend() {
    START_LOG(gettid(), "Call()");
    MPI_Finalize();
    server_println("teardown completed.", CapioCLEngine::get().getWorkflowName(),
                   CAPIO_LOG_SERVER_CLI_LEVEL_INFO, "MPIBackend");
}

const std::set<std::string> MPIBackend::get_nodes() { return nodes; }

void MPIBackend::handshake_servers() {
    START_LOG(gettid(), "call()");

    auto buf = std::unique_ptr<char[]>(new char[MPI_MAX_PROCESSOR_NAME]);
    for (int i = 0; i < n_servers; i += 1) {
        if (i != rank) {
            // TODO: possible deadlock
            MPI_Send(node_name.c_str(), node_name.length(), MPI_CHAR, i, 0, MPI_COMM_WORLD);
            std::fill(buf.get(), buf.get() + MPI_MAX_PROCESSOR_NAME, 0);
            MPI_Recv(buf.get(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, i, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            nodes.emplace(buf.get());
            hostname_to_rank.emplace(buf.get(), i);
            // Without this, rank_to_hostname only ever holds this server's own rank (set in
            // the constructor). Every lookup for a remote rank then goes through
            // operator[] and silently yields an empty string, so replies are addressed to
            // "", send_request() resolves "" through operator[] to rank 0, and the reply is
            // misrouted - typically back to the sender itself, which then handles a reply
            // for a tid it never registered and dereferences a null offset pointer.
            rank_to_hostname.emplace(i, buf.get());
        }
    }
}

RemoteRequest MPIBackend::read_next_request() {
    START_LOG(gettid(), "call()");
    MPI_Status status;
    char *buff = new char[CAPIO_SERVER_REQUEST_MAX_SIZE];
    LOG("initiating a lightweight MPI receive");
    MPI_Request request;
    int received = 0;

    // receive from server
    MPI_Irecv(buff, CAPIO_SERVER_REQUEST_MAX_SIZE, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
              &request);
    struct timespec sleepTime, returnTime;
    sleepTime.tv_sec  = 0;
    sleepTime.tv_nsec = 200000;

    while (!received) {
        MPI_Test(&request, &received, &status);
        nanosleep(&sleepTime, &returnTime);
    }
    int bytes_received;
    MPI_Get_count(&status, MPI_CHAR, &bytes_received);

    LOG("receive completed!");
    return {buff, hostname_of_rank(rank_to_hostname, status.MPI_SOURCE)};
}
void MPIBackend::send_request(const char *message, int message_len, const std::string &target) {
    START_LOG(gettid(), "call(message=%s, message_len=%d, target=%s)", message, message_len,
              target.c_str());
    // operator[] on a miss would insert the unknown target mapped to rank 0 and send the
    // message there, silently and wrongly; fail loudly instead.
    const auto target_entry = hostname_to_rank.find(target);
    if (target_entry == hostname_to_rank.end()) {
        ERR_EXIT("Error: no MPI rank known for target node \"%s\"", target.c_str());
    }
    const auto mpi_target = target_entry->second;
    LOG("MPI_rank for target %s is %c", target.c_str(), mpi_target);

    MPI_Send(message, message_len + 1, MPI_CHAR, mpi_target, 0, MPI_COMM_WORLD);
}

void MPIBackend::send_file(char *shm, long int nbytes, const std::string &target) {
    START_LOG(gettid(), "call(%.50s, %ld, %s)", shm, nbytes, target.c_str());
    int elem_to_snd = 0;
    int dest        = hostname_to_rank[target];
    for (long int k = 0; k < nbytes; k += elem_to_snd) {
        // Compute the maximum amount to send for this chunk
        elem_to_snd = static_cast<int>(std::min(nbytes - k, MPI_MAX_ELEM_COUNT));

        LOG("Sending %d bytes to %d with offset from beginning odf k=%ld", elem_to_snd, dest, k);
        MPI_Isend(shm + k, elem_to_snd, MPI_BYTE, dest, 0, MPI_COMM_WORLD, &req);
        LOG("Sent chunk of %d bytes", elem_to_snd);
    }
}

void MPIBackend::recv_file(char *shm, const std::string &source, long int bytes_expected) {
    START_LOG(gettid(), "call(shm=%ld, source=%s, bytes_expected=%ld)", shm, source.c_str(),
              bytes_expected);
    MPI_Status status;
    int bytes_received = 0, count = 0, source_rank = hostname_to_rank[source];
    LOG("Is buffer valid? %s",
        shm != nullptr ? "yes" : "NO! a nullptr was given to recv_file. this will make MPI crash!");
    for (long int k = 0; k < bytes_expected; k += bytes_received) {

        count = static_cast<int>(std::min(bytes_expected - k, MPI_MAX_ELEM_COUNT));

        LOG("Expected %ld bytes from %s with offset from beginning odf k=%ld", count,
            source.c_str(), k);
        MPI_Recv(shm + k, count, MPI_BYTE, source_rank, 0, MPI_COMM_WORLD, &status);
        LOG("Received chunk");
        MPI_Get_count(&status, MPI_BYTE, &bytes_received);
        LOG("Chunk size is %ld bytes", bytes_received);
    }
}

MPISYNCBackend::MPISYNCBackend(int argc, char *argv[]) : MPIBackend(argc, argv) {
    START_LOG(gettid(), "call()");
    LOG("Wrapped MPI backend with MPISYC backend");
    server_println("initialization completed.", CapioCLEngine::get().getWorkflowName(),
                   CAPIO_LOG_SERVER_CLI_LEVEL_STATUS, "MPISYNCBackend");
}

MPISYNCBackend::~MPISYNCBackend() {
    START_LOG(gettid(), "Call()");
    MPI_Finalize();
    server_println("teardown completed.", CapioCLEngine::get().getWorkflowName(),
                   CAPIO_LOG_SERVER_CLI_LEVEL_INFO, "MPISYNCBackend");
}

RemoteRequest MPISYNCBackend::read_next_request() {
    START_LOG(gettid(), "call()");
    MPI_Status status;
    char *buff = new char[CAPIO_SERVER_REQUEST_MAX_SIZE];
    LOG("initiating a synchronized MPI receive");
    MPI_Recv(buff, CAPIO_SERVER_REQUEST_MAX_SIZE, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
             &status); // receive from server

    LOG("receive completed!");
    return {buff, hostname_of_rank(rank_to_hostname, status.MPI_SOURCE)};
}