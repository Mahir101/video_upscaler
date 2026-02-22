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

enum class Encoder { H264, HEVC, PRORES };

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\n\n⚠️ Interrupt received. Cleaning environment..." << std::endl;
        g_keep_running = false;
        if (!g_temp_dir.empty() && fs::exists(g_temp_dir)) {
            try { fs::remove_all(g_temp_dir); } catch (...) {}
        }
        std::exit(signal);
    }
}

// --- Utilities ---
void run_command(const std::string& cmd, bool quiet = true) {
    std::string final_cmd = cmd;
    if (quiet) final_cmd += " > /dev/null 2>&1";
    int result = std::system(final_cmd.c_str());
    if (result != 0) throw std::runtime_error("Command failed: " + cmd);
}

std::string get_command_output(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) result += buffer.data();
    return result;
}

void print_progress_bar(float progress, int width = 40) {
    std::cout << "\r\033[1;32mProgress: [";
    int pos = width * progress;
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "█";
        else if (i == pos) std::cout << "▓";
        else std::cout << "░";
    }
    std::cout << "] " << int(progress * 100.0) << "%\033[0m" << std::flush;
}

void print_header() {
    std::cout << "\033[1;35m============================================================\033[0m" << std::endl;
    std::cout << "\033[1;33m💎 ULTIMATE VIDEO UPSCALER (Professional Edition)\033[0m" << std::endl;
    std::cout << "   Optimized for Apple Silicon M4" << std::endl;
    std::cout << "\033[1;35m============================================================\033[0m" << std::endl << std::endl;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    print_header();

    // Default Configuration
    std::string input = "";
    std::string output = "output_pro.mp4";
    std::string fps = "60";
    std::string scale = "4";
    Encoder encoder = Encoder::H264;
    bool use_rife = false;

    // Advanced Flag Parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" || arg == "-i") input = argv[++i];
        else if (arg == "--output" || arg == "-o") output = argv[++i];
        else if (arg == "--fps" || arg == "-f") fps = argv[++i];
        else if (arg == "--scale" || arg == "-s") scale = argv[++i];
        else if (arg == "--hevc") encoder = Encoder::HEVC;
        else if (arg == "--prores") { encoder = Encoder::PRORES; output = "output_pro.mov"; }
        else if (arg == "--rife") use_rife = true;
    }

    if (input.empty()) {
        std::cout << "Usage: ./upscaler --input <file> [options]" << std::endl;
        std::cout << "Advanced Options:" << std::endl;
        std::cout << "  --hevc           Use H.265 (Ultra Efficiency)" << std::endl;
        std::cout << "  --prores         Use Apple ProRes (Professional HQ)" << std::endl;
        std::cout << "  --rife           Use RIFE AI interpolation" << std::endl;
        std::cout << "  --fps <val>      Target FPS (default: 60)" << std::endl;
        return 1;
    }

    try {
        g_temp_dir = "temp_ultimate_" + std::to_string(std::time(nullptr));
        fs::create_directories(g_temp_dir + "/lr");
        fs::create_directories(g_temp_dir + "/hr");
        fs::create_directories(g_temp_dir + "/interp");

        std::cout << "� Analyzing Video Stream..." << std::endl;
        std::string fps_cmd = "ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate \"" + input + "\" | head -1";
        std::string orig_fps = get_command_output(fps_cmd);
        if (!orig_fps.empty() && orig_fps.back() == '\n') orig_fps.pop_back();

        // Step 1: Sequential Pipe Extraction
        std::cout << "📽️  Extracting Frames..." << std::endl;
        run_command("ffmpeg -y -i \"" + input + "\" -qscale:v 2 \"" + g_temp_dir + "/lr/f_%07d.png\"");
        
        size_t total_frames = std::distance(fs::directory_iterator(g_temp_dir + "/lr"), fs::directory_iterator{});
        std::cout << "📦 Frames: " << total_frames << " | Source: " << orig_fps << " FPS" << std::endl;

        // Step 2: Parallel AI Upscaling
        std::cout << "🔍 Applying AI Super-Resolution (Real-ESRGAN)..." << std::endl;
        std::string upscale_cmd = "./realesrgan-ncnn-vulkan -i \"" + g_temp_dir + "/lr\" -o \"" + g_temp_dir + "/hr\" -n realesrgan-x4plus -s " + scale + " -t 1024 -f png";
        
        std::atomic<bool> upscale_done(false);
        std::thread monitor([&]() {
            while (!upscale_done) {
                size_t current = std::distance(fs::directory_iterator(g_temp_dir + "/hr"), fs::directory_iterator{});
                print_progress_bar((float)current / total_frames);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            print_progress_bar(1.0); std::cout << std::endl;
        });

        run_command(upscale_cmd);
        upscale_done = true;
        monitor.join();

        // Step 3: Optional AI Interpolation (RIFE)
        std::string final_frame_dir = g_temp_dir + "/hr";
        if (use_rife) {
            std::cout << "✨ Applying AI Fluid Motion (RIFE)..." << std::endl;
            if (fs::exists("./rife-ncnn-vulkan")) {
                run_command("./rife-ncnn-vulkan -i \"" + g_temp_dir + "/hr\" -o \"" + g_temp_dir + "/interp\" -n rife-v4");
                final_frame_dir = g_temp_dir + "/interp";
            } else {
                std::cout << "⚠️ RIFE binary not found. Falling back to high-quality FFmpeg interpolation." << std::endl;
            }
        }

        // Step 4: Final Encode with Hardware Acceleration
        std::cout << "🎬 Master Encoding (Apple Silicon Hardware Engines)..." << std::endl;
        std::string codec_flag, bitrate_flag, pix_fmt = "yuv420p";
        
        switch (encoder) {
            case Encoder::HEVC:
                codec_flag = "hevc_videotoolbox";
                bitrate_flag = "-b:v 15M";
                std::cout << "🚀 Codec: HEVC (H.265)" << std::endl;
                break;
            case Encoder::PRORES:
                codec_flag = "prores_videotoolbox -profile:v 3"; // 422 HQ
                bitrate_flag = "";
                pix_fmt = "yuv422p10le";
                std::cout << "🎥 Codec: Apple ProRes 422 HQ" << std::endl;
                break;
            default:
                codec_flag = "h264_videotoolbox";
                bitrate_flag = "-b:v 25M";
                std::cout << "📽️ Codec: H.264" << std::endl;
                break;
        }

        std::string filter = (final_frame_dir == g_temp_dir + "/interp") ? "" : "-vf \"minterpolate=fps=" + fps + ":mi_mode=mci:mc_mode=aobmc\"";
        
        std::string encode_cmd = "ffmpeg -y -framerate " + (use_rife ? fps : orig_fps) + " -i \"" + final_frame_dir + "/f_%07d.png\" " +
                                filter + " -c:v " + codec_flag + " " + bitrate_flag + " -pix_fmt " + pix_fmt + " -an \"" + g_temp_dir + "/no_audio.mp4\"";
        run_command(encode_cmd);

        // Step 5: Audio Mix
        std::cout << "🔊 Mixing Master Audio..." << std::endl;
        std::string audio_cmd = "ffmpeg -y -i \"" + g_temp_dir + "/no_audio.mp4\" -i \"" + input + "\" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest \"" + output + "\"";
        run_command(audio_cmd);

        fs::remove_all(g_temp_dir);
        std::cout << "\n\033[1;32m✅ SUCCESS! Final Result: " << output << "\033[0m" << std::endl;

    } catch (const std::exception& e) {
        if (!g_temp_dir.empty()) fs::remove_all(g_temp_dir);
        std::cerr << "\033[1;31m❌ Error: " << e.what() << "\033[0m" << std::endl;
        return 1;
    }
    return 0;
}
