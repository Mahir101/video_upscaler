#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <ctime>
#include <csignal>
#include <thread>
#include <chrono>
#include <atomic>
#include <iomanip>

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  #error "No filesystem header found"
#endif

// --- Global State for Signal Handling ---
std::atomic<bool> g_keep_running(true);
std::string g_temp_dir = "";

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\n\n⚠️  Interrupt received (Ctrl+C). Cleaning up and exiting..." << std::endl;
        g_keep_running = false;
        if (!g_temp_dir.empty() && fs::exists(g_temp_dir)) {
            try {
                fs::remove_all(g_temp_dir);
                std::cout << "🧹 Cleaned up: " << g_temp_dir << std::endl;
            } catch (...) {}
        }
        std::exit(signal);
    }
}

// --- Utilities ---
void run_command(const std::string& cmd, bool quiet = true) {
    std::string final_cmd = cmd;
    if (quiet) {
        final_cmd += " > /dev/null 2>&1";
    }
    int result = std::system(final_cmd.c_str());
    if (result != 0) {
        throw std::runtime_error("Command failed with exit code " + std::to_string(result) + ": " + cmd);
    }
}

std::string get_command_output(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

void print_progress_bar(float progress, int width = 40) {
    std::cout << "\r[";
    int pos = width * progress;
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "█";
        else if (i == pos) std::cout << "▓";
        else std::cout << "░";
    }
    std::cout << "] " << int(progress * 100.0) << "% " << std::flush;
}

// --- Heavy Lifting ---
void print_header() {
    std::cout << "\033[1;36m============================================================\033[0m" << std::endl;
    std::cout << "\033[1;33m🎥 VIDEO UPSCALER PRO (M4 Optimized)\033[0m" << std::endl;
    std::cout << "   Created by Md. Mahir Labib" << std::endl;
    std::cout << "   Hardware: Apple Silicon (VideoToolbox Enabled)" << std::endl;
    std::cout << "\033[1;36m============================================================\033[0m" << std::endl << std::endl;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    print_header();

    // Simple Argument Parsing
    std::string input = "";
    std::string output = "output_pro.mp4";
    std::string fps = "60";
    std::string scale = "4";
    std::string model = "realesrgan-x4plus";
    std::string tile = "1024"; // Higher default for M4 to avoid seams
    std::string limit_frames = ""; // All by default
    bool keep_temp = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" || arg == "-i") input = argv[++i];
        else if (arg == "--output" || arg == "-o") output = argv[++i];
        else if (arg == "--fps" || arg == "-f") fps = argv[++i];
        else if (arg == "--scale" || arg == "-s") scale = argv[++i];
        else if (arg == "--model" || arg == "-m") model = argv[++i];
        else if (arg == "--tile" || arg == "-t") tile = argv[++i];
        else if (arg == "--frames" || arg == "-n") limit_frames = argv[++i];
        else if (arg == "--keep-temp") keep_temp = true;
    }

    if (input.empty()) {
        std::cout << "Usage: upscaler --input <file> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  -o, --output <file>  (default: output_pro.mp4)" << std::endl;
        std::cout << "  -f, --fps <val>       (default: 60)" << std::endl;
        std::cout << "  -s, --scale <val>     (default: 4)" << std::endl;
        std::cout << "  -m, --model <name>    (default: realesrgan-x4plus)" << std::endl;
        std::cout << "  -t, --tile <val>      (default: 0 - auto)" << std::endl;
        std::cout << "  -n, --frames <count>  (default: all)" << std::endl;
        return 1;
    }

    try {
        g_temp_dir = "temp_pro_" + std::to_string(std::time(nullptr));
        fs::create_directories(g_temp_dir + "/lr");
        fs::create_directories(g_temp_dir + "/hr");

        std::cout << "🚀 Initializing environment..." << std::endl;
        
        // Get metadata
        std::string fps_cmd = "ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate \"" + input + "\" | head -1";
        std::string orig_fps = get_command_output(fps_cmd);
        if (!orig_fps.empty() && orig_fps.back() == '\n') orig_fps.pop_back();

        // Extract
        std::cout << "📽️  Extracting frames (Hardware accelerated read)..." << std::endl;
        std::string extract_cmd = "ffmpeg -y -i \"" + input + "\" ";
        if (!limit_frames.empty()) extract_cmd += "-frames:v " + limit_frames + " ";
        extract_cmd += "-qscale:v 2 \"" + g_temp_dir + "/lr/f_%07d.png\"";
        run_command(extract_cmd);
        
        size_t total_frames = std::distance(fs::directory_iterator(g_temp_dir + "/lr"), fs::directory_iterator{});
        std::cout << "📦 Total frames to process: " << total_frames << std::endl;

        // Upscale with Progress Monitoring
        std::cout << "🔍 Upscaling with Real-ESRGAN (" << scale << "x, Tile: " << (tile == "0" ? "Auto" : tile) << ")..." << std::endl;
        std::string upscale_cmd = "./realesrgan-ncnn-vulkan -i \"" + g_temp_dir + "/lr\" -o \"" + g_temp_dir + "/hr\" -n " + model + " -s " + scale + " -t " + tile + " -f png";
        
        // Run upscale in a thread so we can monitor progress
        std::atomic<bool> upscale_done(false);
        std::thread monitor([&]() {
            while (!upscale_done) {
                size_t current_frames = std::distance(fs::directory_iterator(g_temp_dir + "/hr"), fs::directory_iterator{});
                float progress = (float)current_frames / total_frames;
                print_progress_bar(progress);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            print_progress_bar(1.0);
            std::cout << std::endl;
        });

        run_command(upscale_cmd);
        upscale_done = true;
        monitor.join();

        // Reassemble with M4 VideoToolbox
        std::cout << "🎬 Reassembling with M4 Hardware Acceleration (VideoToolbox)..." << std::endl;
        // Step 1: Motion Interpolation to target FPS
        std::string ffmpeg_pro = "ffmpeg -y -framerate " + orig_fps + " -i \"" + g_temp_dir + "/hr/f_%07d.png\" "
                                "-vf \"minterpolate=fps=" + fps + ":mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1\" "
                                "-c:v h264_videotoolbox -b:v 20M -pix_fmt yuv420p -an \"" + g_temp_dir + "/no_audio.mp4\"";
        run_command(ffmpeg_pro);

        // Step 2: Add Audio
        std::cout << "🔊 Mixing audio..." << std::endl;
        std::string audio_cmd = "ffmpeg -y -i \"" + g_temp_dir + "/no_audio.mp4\" -i \"" + input + "\" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest \"" + output + "\"";
        run_command(audio_cmd);

        // Cleanup
        if (!keep_temp) {
            fs::remove_all(g_temp_dir);
            g_temp_dir = "";
        } else {
            std::cout << "📁 Keeping temporary files in: " << g_temp_dir << std::endl;
        }

        std::cout << std::endl << "\033[1;32m✅ DONE! Pro Upscale successful.\033[0m" << std::endl;
        std::cout << "🎉 Result saved to: " << output << std::endl;

    } catch (const std::exception& e) {
        if (!g_temp_dir.empty() && !keep_temp) fs::remove_all(g_temp_dir);
        else if (!g_temp_dir.empty() && keep_temp) std::cout << "📁 Keeping temporary files for debugging in: " << g_temp_dir << std::endl;
        std::cerr << "\033[1;31m❌ Error: " << e.what() << "\033[0m" << std::endl;
        return 1;
    }

    return 0;
}
