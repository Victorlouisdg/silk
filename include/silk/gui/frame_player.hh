#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"
#include <iostream>

namespace silk {

/**
 * @brief Adds two side-by-side buttons for going to the previous and next frame.
 *
 * @param frame The current frame.
 * @param num_frames  The number of frames.
 * @return The current frame, which may be different from the input frame if the user pressed a button.
 */
int PrevNextButtons(int frame, int num_frames) {
  if (ImGui::Button("Previous frame")) {
    frame = (frame - 1) % num_frames;  // Apparently % does not make negative numbers positive
    if (frame == -1) {
      frame = num_frames - 1;
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Next frame")) {
    frame = (frame + 1) % num_frames;
  }
  return frame;
}

/**
 * @brief A ImGUI-stype widget for playing through a sequence of frames with Play/Pause and Previous/Next buttons.
 *
 * @param num_frames The number of frames, frame 0 is the start frame and the last frame is num_frames - 1.
 * @return The frame we are currently on.
 */
int FramePlayer(int num_frames) {
  static int frame = 0;  // Initialize to -1 so that the first frame is 0
  static bool paused = true;

  if (!paused) {
    frame = (frame + 1) % num_frames;
  }

  ImGui::Text("Frame %d", frame);

  if (paused) {
    if (ImGui::Button("Play")) {
      paused = false;
    }
    frame = PrevNextButtons(frame, num_frames);
  } else {
    if (ImGui::Button("Pause")) {
      paused = true;
    }
  }
  return frame;
}

}  // namespace silk