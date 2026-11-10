"""Webcam entry point; the logic lives in attention_tracker."""

if __package__:
    from .attention_tracker import runWebcam
else:
    from attention_tracker import runWebcam

if __name__ == "__main__":
    runWebcam()
