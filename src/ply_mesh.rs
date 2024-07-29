use nom::{
    bytes::complete::{tag, take_while},
    sequence::tuple,
    IResult,
};

struct PlyHeader {
    format: String,
    vertex_count: usize,
    face_count: usize,
}

struct Vertex {
    x: f32,
    y: f32,
    z: f32,
}

struct Face {
    indices: Vec<u32>,
}

struct PlyMesh {
    vertices: Vec<Vertex>,
    faces: Vec<Face>,
}

fn parse_header(input: &str) -> IResult<&str, PlyHeader> {
    let (input, _) = tag("ply")(input)?;
    let (input, _) = tag("format ascii 1.0")(input)?;
    Ok((
        input,
        PlyHeader {
            format: "ascii".to_string(),
            vertex_count: 100,
            face_count: 50,
        },
    ))
}

fn parse_vertex(input: &str) -> IResult<&str, Vertex> {
    let (input, (x, y, z)) = tuple((parse_float, parse_float, parse_float))(input)?;
    Ok((input, Vertex { x, y, z }))
}

fn parse_float(input: &str) -> IResult<&str, f32> {
    let (input, number) = take_while(|c: char| c.is_digit(10) || c == '.' || c == '-')(input)?;
    Ok((input, number.parse().unwrap()))
}
