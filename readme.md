# Temperature Monitoring with Colorful ANSI Output

This Rust project reads the temperature and fan speeds of a compatible Dell system running Linux using the `i8kctl` command and outputs the information with colorful ANSI escape codes.

## Features

- Interpolates colors based on temperature ranges smoothly with the amazing Oklab colour space.
- Outputs temperature and fan speeds with ANSI escape codes for colorful terminal display.
- Supports diffing ANSI strings to only update changes.

## Temperature Color Scale

The temperature color scale is designed to visually represent different temperature levels. Below are examples of temperature ranges displayed with their corresponding foreground and background colors:
- <span style="color: rgb(22, 114, 22); background-color: rgb(0, 0, 0);">30°C</span>
- <span style="color: rgb(48, 215, 48); background-color: rgb(0, 0, 0);">40°C</span>
- <span style="color: rgb(255, 255, 0); background-color: rgb(0, 0, 0);">45°C</span>
- <span style="color: rgb(255, 165, 0); background-color: rgb(0, 0, 0);">50°C</span>
- <span style="color: rgb(255, 0, 0); background-color: rgb(0, 0, 0);">60°C</span>
- <span style="color: rgb(255, 53, 53); background-color: rgb(0, 0, 0);">65°C</span>
- <span style="color: rgb(255, 132, 48); background-color: rgb(139, 0, 0);">80°C</span>

i let chatgpt write this based on my code. honestly it was too stupid. just writing it myself would have been faster lmao
