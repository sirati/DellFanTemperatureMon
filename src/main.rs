use std::{fmt, io};
use std::cmp::PartialEq;
use std::fmt::{Debug, Formatter};
use std::io::{BufRead, StdinLock, Write};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::Receiver;
use std::thread::sleep;

use arrayvec::ArrayVec;
use crossterm::terminal;
use once_cell::unsync::Lazy;
use scopeguard::defer;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Copy)]
enum Color {
    Rgb(Rgb),
    Oklab(Oklab),
}

impl Color {
    fn to_rgb(&self) -> Rgb {
        match self {
            Color::Rgb(rgb) => *rgb,
            Color::Oklab(oklab) => oklab.to_rgb(),
        }
    }

    fn to_oklab(&self) -> Oklab {
        match self {
            Color::Rgb(rgb) => rgb.to_oklab(),
            Color::Oklab(oklab) => *oklab,
        }
    }
}

impl PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Color::Rgb(a), Color::Rgb(b)) => a == b,
            (Color::Oklab(a), Color::Oklab(b)) => a == b,
            (Color::Rgb(a), _) => a == &other.to_rgb(),
            (_, Color::Rgb(b)) => b == &self.to_rgb(),
        }
    }
}

// Define newtypes for RGB and Oklab color spaces
#[derive(Debug, Clone, Copy, PartialEq)]
struct Rgb(u8, u8, u8);

#[derive(Debug, Clone, Copy, PartialEq)]
struct Oklab(f64, f64, f64);

impl Rgb {
    fn to_oklab(&self) -> Oklab {
        let r = self.0 as f64 / 255.0;
        let g = self.1 as f64 / 255.0;
        let b = self.2 as f64 / 255.0;

        let r_linear = if r >= 0.04045 { ((r + 0.055) / 1.055).powf(2.4) } else { r / 12.92 };
        let g_linear = if g >= 0.04045 { ((g + 0.055) / 1.055).powf(2.4) } else { g / 12.92 };
        let b_linear = if b >= 0.04045 { ((b + 0.055) / 1.055).powf(2.4) } else { b / 12.92 };

        let l = 0.4122214708 * r_linear + 0.5363325363 * g_linear + 0.0514459929 * b_linear;
        let m = 0.2119034982 * r_linear + 0.6806995451 * g_linear + 0.1073969566 * b_linear;
        let s = 0.0883024619 * r_linear + 0.2817188376 * g_linear + 0.6299787005 * b_linear;

        let l_cbrt = cbrt(l);
        let m_cbrt = cbrt(m);
        let s_cbrt = cbrt(s);

        #[allow(non_snake_case)]
        let L = l_cbrt * 0.2104542553 + m_cbrt * 0.7936177850 - s_cbrt * 0.0040720468;
        let a = l_cbrt * 1.9779984951 - m_cbrt * 2.4285922050 + s_cbrt * 0.4505937099;
        let b = l_cbrt * 0.0259040371 + m_cbrt * 0.7827717662 - s_cbrt * 0.8086757660;

        Oklab(L, a, b)
    }

    fn is_max_diff_1(self, other: Rgb) -> bool {
        let r_diff = (self.0 as i16 - other.0 as i16).abs();
        let g_diff = (self.1 as i16 - other.1 as i16).abs();
        let b_diff = (self.2 as i16 - other.2 as i16).abs();
        r_diff <= 1 && g_diff <= 1 && b_diff <= 1
    }

    fn interpolate_colors(&self, target: f64, floor: f64, ceiling: f64, to_colour: Rgb) -> Rgb {
        self.to_oklab().interpolate_colors(target, floor, ceiling, to_colour.to_oklab()).to_rgb()
    }
}

impl Oklab {
    fn to_rgb(&self) -> Rgb {
        let l = self.0 + self.1 * 0.3963377774 + self.2 * 0.2158037573;
        let m = self.0 + self.1 * -0.1055613458 + self.2 * -0.0638541728;
        let s = self.0 + self.1 * -0.0894841775 + self.2 * -1.2914855480;

        let l_cubed = l.powf(3.0);
        let m_cubed = m.powf(3.0);
        let s_cubed = s.powf(3.0);

        let r = l_cubed * 4.0767416621 + m_cubed * -3.3077115913 + s_cubed * 0.2309699292;
        let g = l_cubed * -1.2684380046 + m_cubed * 2.6097574011 + s_cubed * -0.3413193965;
        let b = l_cubed * -0.0041960863 + m_cubed * -0.7034186147 + s_cubed * 1.7076147010;

        let r = if r >= 0.0031308 {
            (1.055 * r.powf(1.0 / 2.4) - 0.055) * 255.0
        } else {
            12.92 * r * 255.0
        };

        let g = if g >= 0.0031308 {
            (1.055 * g.powf(1.0 / 2.4) - 0.055) * 255.0
        } else {
            12.92 * g * 255.0
        };

        let b = if b >= 0.0031308 {
            (1.055 * b.powf(1.0 / 2.4) - 0.055) * 255.0
        } else {
            12.92 * b * 255.0
        };

        Rgb(
            clamp(r, 0.0, 255.0) as u8,
            clamp(g, 0.0, 255.0) as u8,
            clamp(b, 0.0, 255.0) as u8,
        )
    }

    fn interpolate_colors(&self, target: f64, floor: f64, ceiling: f64, to_colour: Oklab) -> Oklab {
        let (from, to) = (self, to_colour);
        let t_ratio = (target - floor) / (ceiling - floor);
        Oklab(
            from.0 + t_ratio * (to.0 - from.0),
            from.1 + t_ratio * (to.1 - from.1),
            from.2 + t_ratio * (to.2 - from.2),
        )
    }
}

#[derive(Debug, Clone)]
enum AnsiTemplatePart {
    ControlSequenceConst(&'static str, isize, bool), // Control sequence, x_pos, is_absolute
    Foreground(),
    Background(),
    AsciiConst(&'static str),
    AsciiVariable(usize),
}


#[derive(Debug, Clone)]
struct AnsiStringTemplate<const LENGTH: usize, const COLOURS: usize, const VARIABLES: usize> {
    sequences: [AnsiTemplatePart; LENGTH],
}

impl<const LENGTH: usize, const COLOURS: usize, const VARIABLES: usize> AnsiStringTemplate<LENGTH, COLOURS, VARIABLES> {
    fn new(sequences: [AnsiTemplatePart; LENGTH]) -> Self {
        let mut color_count = 0;
        let mut variable_count = 0;
        for sequence in &sequences {
            match sequence {
                AnsiTemplatePart::ControlSequenceConst(_, _, _) => (),
                AnsiTemplatePart::Foreground() => color_count += 1,
                AnsiTemplatePart::Background() => color_count += 1,
                AnsiTemplatePart::AsciiConst(_) => (),
                AnsiTemplatePart::AsciiVariable(_) => variable_count += 1,
            }
        }
        assert_eq!(color_count, COLOURS);
        assert_eq!(variable_count, VARIABLES);
        Self {
            sequences,
        }
    }

    fn with(&self, colors: [Color; COLOURS], variable_values: [&dyn fmt::Display; VARIABLES]) -> AnsiString<LENGTH, COLOURS, VARIABLES> {
        AnsiDisplay {
            template: self.clone(),
            colors,
            variables: variable_values.iter()
                .map(|display| AnsiVariable::Display(*display))
                .collect::<ArrayVec<AnsiVariable, VARIABLES>>().into_inner().unwrap_or_else(|_| panic!("Expected {} variables", VARIABLES)),
        }.to_ansi_string()
    }
}

#[derive(Clone)]
enum AnsiVariable<'a> {
    Display(&'a dyn fmt::Display),
    String(String),
}


struct AnsiDisplay<'a, const LENGTH: usize, const COLOURS: usize, const VARIABLES: usize> {
    template: AnsiStringTemplate<LENGTH, COLOURS, VARIABLES>,
    colors: [Color; COLOURS],
    variables: [AnsiVariable<'a>; VARIABLES],
}

impl<'a, const LENGTH: usize, const COLOURS: usize, const VARIABLES: usize>
AnsiDisplay<'a, LENGTH, COLOURS, VARIABLES> {
    fn to_ansi_string(&self) -> AnsiString<LENGTH, COLOURS, VARIABLES> {
        let variables_as_strings: ArrayVec<String, VARIABLES> = self.variables.iter().zip(self.template.sequences.iter().filter_map(|part| match part {
            AnsiTemplatePart::AsciiVariable(length) => Some(length.clone()),
            _ => None,
        })).map(|(variable, length)| {
            let str: String = match variable {
                AnsiVariable::Display(display) => display.to_string(),
                AnsiVariable::String(string) => string.clone(),
            };
            let (count, index) = str.grapheme_indices(true).take(length).enumerate()
                .filter(|(count, _)| *count <= length)
                .max_by_key(|(_, (index, _))| *index)
                .map(|(count, (idx, _))| (count + 1, idx + 1))
                .unwrap_or((0, 0));

            let r =if count == length {
                str[..index].to_string()
            } else {
                " ".repeat(length - count) + str.as_str()
            };
            r
        }).collect();
        AnsiString {
            template: self.template.clone(),
            colors: self.colors,
            variables: variables_as_strings.into_inner().unwrap(),
        }
    }
}


struct AnsiString<const LENGTH: usize, const COLOURS: usize, const variables: usize> {
    template: AnsiStringTemplate<LENGTH, COLOURS, variables>,
    colors: [Color; COLOURS],
    variables: [String; variables],
}



impl<const LENGTH: usize, const COLOURS: usize, const VARIABLES: usize> fmt::Display for
AnsiString<LENGTH, COLOURS, VARIABLES> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut color_index = 0;
        let mut variable_index = 0;
        let mut x_pos = 0;
        for sequence in &self.template.sequences {
            match sequence {
                AnsiTemplatePart::ControlSequenceConst(sequence, x_pos_new, is_absolute) => {
                    x_pos = if *is_absolute { *x_pos_new } else { x_pos + *x_pos_new };
                    write!(f, "{}", sequence)?;
                }
                AnsiTemplatePart::Foreground() | AnsiTemplatePart::Background() => {
                    let rgb = self.colors[color_index].to_rgb();
                    match sequence {
                        AnsiTemplatePart::Foreground() =>
                            write!(f, "\x1b[38;2;{};{};{}m", rgb.0, rgb.1, rgb.2)?,
                        AnsiTemplatePart::Background() =>
                            write!(f, "\x1b[48;2;{};{};{}m", rgb.0, rgb.1, rgb.2)?,
                        _ => {}
                    };
                    color_index += 1;
                }
                AnsiTemplatePart::AsciiConst(ascii) => {
                    x_pos += ascii.graphemes(true).count() as isize;
                    write!(f, "{}", ascii)?;
                }
                AnsiTemplatePart::AsciiVariable(length) => {
                    x_pos += *length as isize;
                    write!(f, "{}", self.variables[variable_index])?;
                    variable_index += 1;
                }
            }
        }
        //move cursor back to start of line, x_pos is how much we moved to the right, so move x_pos left
        write!(f, "\x1b[{}D", x_pos)?;

        Ok(())
    
    }
}

struct AnsiStringDiff<'a, const LENGTH: usize, const COLOURS: usize, const VARIABLES: usize> {
    old: &'a AnsiString<LENGTH, COLOURS, VARIABLES>,
    new: &'a AnsiString<LENGTH, COLOURS, VARIABLES>,
}

impl<const LENGTH: usize, const COLOURS: usize, const VARIABLES: usize> fmt::Display for
AnsiStringDiff<'_, LENGTH, COLOURS, VARIABLES> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut color_index = 0;
        let mut variable_index = 0;
        let mut last_x_pos = 0;
        let mut x_pos = 0;
        let mut out: Vec<(bool, String)> = Vec::new();

        for new_sequence in self.new.template.sequences.iter() {
            match new_sequence {
                AnsiTemplatePart::ControlSequenceConst(new_sequence, x_pos_new, is_absolute) => {
                    x_pos = if *is_absolute { *x_pos_new } else { x_pos + *x_pos_new };
                    out.push((false, new_sequence.to_string()));
                }
                AnsiTemplatePart::Background() | AnsiTemplatePart::Foreground() =>  {
                    let rgb = self.new.colors[color_index].to_rgb();
                    out.push((false, match new_sequence {
                        AnsiTemplatePart::Foreground() =>
                            format!("\x1b[38;2;{};{};{}m", rgb.0, rgb.1, rgb.2),
                        AnsiTemplatePart::Background() =>
                            format!("\x1b[48;2;{};{};{}m", rgb.0, rgb.1, rgb.2),
                        _ => "".to_string()
                    }));
                    color_index += 1;
                }
                AnsiTemplatePart::AsciiConst(ascii) => {
                    x_pos += ascii.graphemes(true).count() as isize;
                }
                AnsiTemplatePart::AsciiVariable(length) => {
                    //lump from last_x_pos to x_pos
                    //write!(f, "\x1b[{}D{}", x_pos, self.new.variables[variable_index])?;
                    //lump to the right
                    let new_x_pos = x_pos + *length as isize;
                    if self.new.variables[variable_index] != self.old.variables[variable_index] {
                        out.push((true, format!("\x1b[{}C{}", x_pos - last_x_pos, self.new.variables[variable_index])));
                        last_x_pos = new_x_pos;
                    }
                    x_pos = new_x_pos;
                    variable_index += 1;
                }
            }
        }

        if out.iter().any(|(visible, _)| *visible) {
            //move cursor back
            //out.push((false, format!( "\x1b[{}D", x_pos)));

            let out = out.into_iter().map(|(_, s)| s).collect::<String>();
            f.write_str(&out)?;
            write!(f, "\x1b[{}D", x_pos)?;
            //write!(f, "\x1b[u",)?;
        }

        Ok(())
        //move cursor back to start of line, x_pos is how much we
    }
}


const DARK_RED: Rgb = Rgb(139, 0, 0);
const BACKGROUND_TEMP: [u8; 2] = [65, 80];

const COLORS: [(u8, Rgb); 7] = [
    (30, Rgb(22, 114, 22)),    // dark green
    (40, Rgb(48, 215, 48)),    // light green
    (45, Rgb(255, 255, 0)),  // yellow
    (50, Rgb(255, 165, 0)),  // orange
    (60, Rgb(255, 0, 0)),    // bright red
    (BACKGROUND_TEMP[0], Rgb(255, 53, 53)),    // bright light red
    (BACKGROUND_TEMP[1], Rgb(255, 132, 48)),    // orange
];



fn cbrt(x: f64) -> f64 {
    x.powf(1.0 / 3.0)
}

fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

fn get_foreground_color(temp: f64) -> Rgb {
    let mut lower_bound = COLORS.first().unwrap();
    let mut upper_bound = COLORS.last().unwrap();


    for colour in &COLORS {
        let bound = colour.0 as f64;
        if temp >= bound {
            lower_bound = colour;
        } else {
            upper_bound = colour;
            break;
        }
    }

    if lower_bound.0 == upper_bound.0 {
        return lower_bound.1;
    }

    lower_bound.1.interpolate_colors(temp, lower_bound.0 as f64, upper_bound.0 as f64, upper_bound.1)
}

fn get_background_color(temp: f64) -> Rgb {
    let background = get_console_background_color().unwrap_or(Rgb(0, 0, 0));
    if temp <= 65.0 {
        background
    } else if temp >= 80.0 {
        DARK_RED
    } else {
        background.interpolate_colors(temp, 65.0, 80.0, DARK_RED)
    }
}


fn test_rgb_oklab_rgb(rgb: Rgb) {
    let oklab = rgb.to_oklab();
    let rgb2 = oklab.to_rgb();

    println!("Original RGB: {:?}", rgb);
    println!("Converted RGB: {:?}", rgb2);

    //check if difference is less than 1
    if rgb.is_max_diff_1(rgb2) {
        println!("Conversion successful: True");
    } else {
        println!("Conversion successful: False");
    }
    println!();
}

fn get_console_background_color() -> io::Result<Rgb> {

    let mut result = None;
    {
        // Escape sequence to query the color and cursor position

        //ask for background color, ask for cursor
        let query = b"\x1b]4;0;?\x07\x1b[6n";
        let mut stdout = io::stdout();


        stdout.write_all(query)?;
        stdout.flush()?;

        let mut buffer = Vec::new();

        // Flags and counters to handle the response
        let cursor_response_prefix = "\x1b[";
        let mut cursor_response_detected = false;
        let mut got_answer = false;
        let mut byte = 0u8;
        
        fn read_byte(into: &mut u8) -> bool {
            STDIN_CHANNEL.get().map_or_else(|| false, |mutex| {
                mutex.lock().map_or_else(
                    |_| false,
                    |receiver| {
                    receiver.recv_timeout(std::time::Duration::from_millis(100))
                        .map(|b| {
                            *into = b;
                            true
                        }).unwrap_or(false)
                })
            })
        }
        
        

        while read_byte(&mut byte) || (byte!= 0 && !got_answer) {
            if !got_answer && byte == 27 {
                got_answer = true;
            }/* else if !got_answer {
                print!("{}={}", byte, byte as char);
                continue;
            }*/

            buffer.push(byte);

            // Check if we detected the cursor position response
            if !cursor_response_detected && buffer.len() == cursor_response_prefix.len() {
                if &buffer == cursor_response_prefix.as_bytes() {
                    cursor_response_detected = true;
                }
            }

            // Count BEL characters and decide when to stop
            if byte == 0x07 { // BEL character
                result = parse_rgb(&String::from_utf8_lossy(&buffer));
                buffer.clear();
            } else if cursor_response_detected && byte == 82 {
                break
            }
        }
    }
    return Ok(result.unwrap_or(Rgb(0, 0, 0)));
}

fn parse_rgb(response: &str) -> Option<Rgb> {
    // Look for the RGB part in the response
    let prefix = "\x1b]4;0;rgb:";
    let suffix = "\x07";

    if let Some(start) = response.find(prefix) {
        if let Some(end) = response.find(suffix) {
            let rgb_part = &response[start + prefix.len()..end];
            let parts: Vec<&str> = rgb_part.split('/').collect();

            if parts.len() == 3 {


                if let (Some(r), Some(g), Some(b)) =
                    if parts[0].len() == 2 {
                        (
                            u8::from_str_radix(&parts[0][..2], 16).ok(),
                            u8::from_str_radix(&parts[1][..2], 16).ok(),
                            u8::from_str_radix(&parts[2][..2], 16).ok(),
                        )
                    } else if parts[0].len() == 4 {
                        (
                            u16::from_str_radix(&parts[0][..4], 16).map(|x| (x >> 8) as u8).ok(),
                            u16::from_str_radix(&parts[1][..4], 16).map(|x| (x >> 8) as u8).ok(),
                            u16::from_str_radix(&parts[2][..4], 16).map(|x| (x >> 8) as u8).ok(),
                        )
                    } else {
                        (None, None, None)
                    }{

                    return Some(Rgb(r, g, b));
                }
            }
        }
    }

    None
}

struct I8kctlOutput {
    temperature: f64,
    left_fan_speed: u32,
    right_fan_speed: u32,
    left_fan_set_speed: u32,
    right_fan_set_speed: u32,
}



//make oncecess out of template
const OUTPUT_TEMPLATE: Lazy<AnsiStringTemplate<14, 2, 5>> = Lazy::new(|| {
    AnsiStringTemplate::new([
                                AnsiTemplatePart::AsciiConst("Temp: "),
                                AnsiTemplatePart::Foreground(),
                                AnsiTemplatePart::Background(),
                                AnsiTemplatePart::AsciiVariable(6), //"°C"),
                                AnsiTemplatePart::ControlSequenceConst("\x1b[0m", 0, false),
                                AnsiTemplatePart::AsciiConst(" - Left Fan "),
                                AnsiTemplatePart::AsciiVariable(1),
                                AnsiTemplatePart::AsciiConst("->"),
                                AnsiTemplatePart::AsciiVariable(5),
                                AnsiTemplatePart::AsciiConst(" rpm  Right Fan "),
                                AnsiTemplatePart::AsciiVariable(1),
                                AnsiTemplatePart::AsciiConst("->"),
                                AnsiTemplatePart::AsciiVariable(5),
                                AnsiTemplatePart::AsciiConst(" rpm"),
                            ])
});

impl I8kctlOutput {
    fn from_command() -> Result<Self, String> {
        // Execute the i8kctl command
        let output = Command::new("i8kctl")
            .output()
            .map_err(|e| format!("Failed to execute command: {}", e))?;

        // Convert output to a string
        let output_str = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = output_str.trim().split_whitespace().collect();

        // Check if the output has the expected number of parts
        if parts.len() < 10 {
            return Err("Unexpected output format from i8kctl".to_string());
        }

        // Parse the values from the output
        let temperature = parts[3].parse().map_err(|e| format!("Failed to parse temperature: {}", e))?;
        let left_fan_speed = parts[4].parse().map_err(|e| format!("Failed to parse left fan speed: {}", e))?;
        let right_fan_speed = parts[5].parse().map_err(|e| format!("Failed to parse right fan speed: {}", e))?;
        let left_fan_set_speed = parts[6].parse().map_err(|e| format!("Failed to parse left fan set speed: {}", e))?;
        let right_fan_set_speed = parts[7].parse().map_err(|e| format!("Failed to parse right fan set speed: {}", e))?;


        Ok(Self {
            temperature,
            left_fan_speed,
            right_fan_speed,
            left_fan_set_speed,
            right_fan_set_speed,
        })
    }

    fn to_ansi_string(&self) -> AnsiString<14, 2, 5> {

        let temp_st = format!("{:.1}°C", self.temperature);
        OUTPUT_TEMPLATE.with(
            [Color::Rgb(get_foreground_color(self.temperature)),
                Color::Rgb(get_background_color(self.temperature))],
            [ &temp_st,
                &self.left_fan_speed,
                &self.left_fan_set_speed,
                &self.right_fan_speed,
                &self.right_fan_set_speed]
        )
    }

    fn print_ansi(&self) {
        println!("{}", self.to_ansi_string());
    }


}

static SHUTDOWN: AtomicBool = AtomicBool::new(false);
static SHUTDOWN_STDIN: AtomicBool = AtomicBool::new(false);



static STDIN_CHANNEL: OnceLock<Mutex<Receiver<u8>>> = OnceLock::new();

fn main() {
    defer!(SHUTDOWN_STDIN.store(true, Relaxed));
    terminal::enable_raw_mode().unwrap();
    defer!(_ = terminal::disable_raw_mode());
    //create a new thread that listens to stdin one byte at a time
    //inter thread mpsc channel
    let (sender, receiver) = std::sync::mpsc::channel::<u8>();
    STDIN_CHANNEL.set(Mutex::new(receiver)).unwrap();


    _ = std::thread::spawn(move || {
        defer!(SHUTDOWN.store(true, Relaxed));
        //crossterm enable raw mode
        let stdin = io::stdin();
        let mut handle = stdin.lock();
        let mut out: &[u8] = &[];
        let mut buffer = Vec::new();

        fn fill_buf<'a>(handle: &'a mut StdinLock, out: &mut &'a [u8]) -> bool {
            let result = match handle.fill_buf() {
                Ok(to_out) => {
                    *out = to_out;
                    true
                },
                Err(_) => false
            };
            result
        }



        while !SHUTDOWN_STDIN.load(Relaxed) && fill_buf(&mut handle, &mut out) {
            for byte in out {
                if SHUTDOWN_STDIN.load(Relaxed) {
                    return;
                }

                buffer.push(*byte);
                if *byte == 17/*CTRL Q*/ || *byte == 3/*CTRL C*/ {
                    SHUTDOWN.store(true, Relaxed);
                } else {
                    match sender.send(*byte) {
                        Ok(x) => x,
                        Err(err) => {
                            if SHUTDOWN.load(Relaxed) || SHUTDOWN_STDIN.load(Relaxed) {
                                return;
                            }
                            //move a line down and print the error and then back
                            print!("\x1b[1BError: Failed to send byte to main thread: {}\x1b[1A", err);
                            return;
                        }
                    };
                }
            }
            let len = out.len();
            out = &[]; // we need to burrow this reference as it shares burrow lifetime with handle
            handle.consume(len);
        }
    });


    //enable conceal input and disable cursor
    print!(/*"\x1b[8m*/"\x1b[?25l");
    //defer reset styles and enable cursor
    defer!(print!("\x1b[0m\x1b[?25h"));




    // Test with corner cases
/*    test_rgb_oklab_rgb(Rgb(0, 0, 0));         // black
    test_rgb_oklab_rgb(Rgb(255, 255, 255));   // white
    test_rgb_oklab_rgb(Rgb(255, 0, 0));       // red
    test_rgb_oklab_rgb(Rgb(0, 255, 0));       // green
    test_rgb_oklab_rgb(Rgb(0, 0, 255));       // blue
    test_rgb_oklab_rgb(Rgb(255, 255, 0));     // yellow
    test_rgb_oklab_rgb(Rgb(0, 255, 255));     // cyan
    test_rgb_oklab_rgb(Rgb(255, 0, 255));     // magenta
    test_rgb_oklab_rgb(Rgb(128, 128, 128));   // gray
    test_rgb_oklab_rgb(Rgb(128, 0, 0));       // dark red
    test_rgb_oklab_rgb(Rgb(0, 128, 0));       // dark green
    test_rgb_oklab_rgb(Rgb(0, 0, 128));       // dark blue
    test_rgb_oklab_rgb(Rgb(128, 128, 0));     // olive
    test_rgb_oklab_rgb(Rgb(0, 128, 128));     // teal
    test_rgb_oklab_rgb(Rgb(128, 0, 128));     // purple

    println!();
    println!();*/

    print!("\x1b[s");
    let mut last_output = I8kctlOutput::from_command()
        .expect("Failed to get initial i8kctl output")
        .to_ansi_string();
    print!("{}", last_output);

    let mut shutting_down = false;
    loop {
        for i in 0..20 {
            if !SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
                sleep(std::time::Duration::from_millis(200));
            }
            let output = I8kctlOutput::from_command()
                .expect("Failed to get i8kctl output");
            let current_output = output.to_ansi_string();

            print!("\x1b[u\x1b[s"); //restore saved cursor position, then save again

            if i == 19 || shutting_down {
                //clear current line
                print!("\x1b[2K");
                print!("{}", current_output);
            } else {
                print!("{}", AnsiStringDiff {
                    old: &last_output,
                    new: &current_output,
                });
            }
            last_output = current_output;
            if SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
                if shutting_down {
                    println!();
                    return;
                }
                shutting_down = true;
            }
        }
    }

}
