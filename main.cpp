#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  #error "No filesystem header found"
#endif

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <ctime>

void run_command(const std::string& cmd) {
    std::cout << "Executing: " << cmd << std::endl;
    int result = std::system((cmd + " 2>/tmp/upscaler_error.log").c_str());
    if (result != 0) {
        std::string err_msg = "Command failed with exit code " + std::to_string(result);
        std::FILE* err_file = std::fopen("/tmp/upscaler_error.log", "r");
        if (err_file) {
            char buffer[256];
            while (std::fgets(buffer, sizeof(buffer), err_file)) {
                std::cerr << "  [stderr]: " << buffer;
            }
            std::fclose(err_file);
        }
        throw std::runtime_error(err_msg + ": " + cmd);
    }
}

std::string get_command_output(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

void print_header() {
    std::cout << "============================================================" << std::endl;
    std::cout << "🎥 HIGH-QUALITY VIDEO UPSCALER" << std::endl;
    std::cout << "   Created by Md. Mahir Labib" << std::endl;
    std::cout << "   C++ Orchestrator: Real-ESRGAN + 60 FPS" << std::endl;
    std::cout << "============================================================" << std::endl << std::endl;
}

int main(int argc, char* argv[]) {
    print_header();

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <input_video> [output_video] [target_fps] [scale]" << std::endl;
        return 1;
    }

    std::string input_video = argv[1];
    std::string output_video = (argc > 2) ? argv[2] : "video-upscaled.mp4";
    std::string target_fps = (argc > 3) ? argv[3] : "60";
    std::string scale = (argc > 4) ? argv[4] : "4";

    if (!fs::exists(input_video)) {
        std::cerr << "❌ Input file not found: " << input_video << std::endl;
        return 1;
    }

    try {
        // Create temp directory
        std::string temp_dir = "temp_upscale_" + std::to_string(std::time(nullptr));
        fs::create_directories(temp_dir + "/lr_frames");
        fs::create_directories(temp_dir + "/hr_frames");

        std::cout << "📁 Using temp directory: " << temp_dir << std::endl;

        // Get original FPS
        std::string fps_cmd = "ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate \"" + input_video + "\" | head -1";
        std::string original_fps = get_command_output(fps_cmd);
        // Remove newline
        if (!original_fps.empty() && original_fps.back() == '\n') original_fps.pop_back();
        std::cout << "📽️ Original video FPS: " << original_fps << std::endl;

        // Extract frames
        std::cout << std::endl << "📽️ Extracting frames from video..." << std::endl;
        run_command("ffmpeg -y -i \"" + input_video + "\" -qscale:v 2 \"" + temp_dir + "/lr_frames/frame_%07d.png\" 2>/dev/null");
        
        // Upscale
        std::cout << std::endl << "🔍 Upscaling frames with Real-ESRGAN (" << scale << "x)..." << std::endl;
        run_command("./realesrgan-ncnn-vulkan -i \"" + temp_dir + "/lr_frames\" -o \"" + temp_dir + "/hr_frames\" -n realesrgan-x4plus -s " + scale + " -f png");

        // Interpolate and Create Result
        bool is_video = (input_video.find(".mp4") != std::string::npos || 
                         input_video.find(".mkv") != std::string::npos || 
                         input_video.find(".mov") != std::string::npos || 
                         input_video.find(".avi") != std::string::npos);

        if (is_video) {
            std::cout << std::endl << "🎬 Creating smooth " << target_fps << " FPS video with motion interpolation..." << std::endl;
            std::string ffmpeg_cmd = "ffmpeg -y -framerate " + original_fps + " -i \"" + temp_dir + "/hr_frames/frame_%07d.png\" "
                                    "-vf \"minterpolate=fps=" + target_fps + ":mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1\" "
                                    "-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -an \"" + temp_dir + "/temp_no_audio.mp4\" 2>/dev/null";
            run_command(ffmpeg_cmd);

            // Add Audio
            std::cout << std::endl << "🔊 Adding audio from original video..." << std::endl;
            std::string audio_cmd = "ffmpeg -y -i \"" + temp_dir + "/temp_no_audio.mp4\" -i \"" + input_video + "\" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest \"" + output_video + "\" 2>/dev/null";
            run_command(audio_cmd);
        } else {
            // It's an image
            std::cout << std::endl << "🖼️ Saving upscaled image..." << std::endl;
            // Get the first frame from hr_frames
            std::string hr_frame;
            for (const auto& entry : fs::directory_iterator(temp_dir + "/hr_frames")) {
                hr_frame = entry.path().string();
                break;
            }
            if (!hr_frame.empty()) {
                fs::copy_file(hr_frame, output_video, fs::copy_options::overwrite_existing);
            } else {
                throw std::runtime_error("No upscaled frames found in " + temp_dir + "/hr_frames");
            }
        }

        // Cleanup
        std::cout << std::endl << "🧹 Cleaning up temporary files..." << std::endl;
        fs::remove_all(temp_dir);

        std::cout << std::endl << "============================================================" << std::endl;
        std::cout << "✅ SUCCESS! Output saved to: " << output_video << std::endl;
        std::cout << "============================================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
