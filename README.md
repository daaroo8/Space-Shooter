
# Space-Shooter

Space-Shooter is a space shooter game controlled by facial gestures. Move your ship by squinting your eyes and fire by opening your mouth. The game leverages computer vision to detect facial expressions, offering a unique gameplay experience.

## Requirements

To run the game, you will need to install the following Python libraries:

- `cv2` (OpenCV)
- `mediapipe`
- `numpy`

## Installation

Follow these steps to get the game running on your local machine:

1. Clone the repository:

   ```bash
   git clone https://github.com/daaroo8/Space-Shooter.git
   cd Space-Shooter
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the necessary libraries:

   ```bash
   pip install opencv-python mediapipe numpy
   ```

4. Run the game:

   ```bash
   python space_shooter.py  # Or the appropriate file to start the game
   ```

## How it works

- The game uses facial gestures for controls:
  - **Squint your eyes** to move the spaceship.
  - **Open your mouth** to shoot.

The game is a work in progress, and graphics will be improved in future updates.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
