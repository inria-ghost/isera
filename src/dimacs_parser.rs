// Copyright 2024 Isera Developers

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use isera::basetypes::*;
use petgraph::graph::*;
use std::fs;

pub fn parsed_graph<NUM: CloneableNum>(
    filename: String,
) -> (
    DiGraph<u32, CustomEdgeIndices<NUM>>,
    Vec<(usize, NUM)>,
    Vec<(usize, NUM)>,
)
where
    <NUM as std::str::FromStr>::Err: std::fmt::Debug,
{
    let contents = fs::read_to_string(filename).expect("Should have been able to read the file");
    let mut count = 0;
    let mut graph = DiGraph::<u32, CustomEdgeIndices<NUM>>::new();
    let mut sources: Vec<(usize, NUM)> = vec![];
    let mut sinks: Vec<(usize, NUM)> = vec![];

    contents.lines().for_each(|x| {
        if x.chars().nth(0) == Some('p') {
            let line = x.split(' ').collect::<Vec<&str>>()[1..].to_vec();
            for i in 0..line[1].parse::<u32>().unwrap() + 1 {
                graph.add_node(i);
            }
        };
        if x.chars().nth(0) == Some('n') {
            let line = x.split(' ').collect::<Vec<&str>>()[1..].to_vec();
            if line[1].parse::<NUM>().unwrap().is_negative() {
                sinks.push((
                    line[0].parse::<usize>().unwrap(),
                    line[1].parse::<NUM>().unwrap(),
                ));
            } else {
                sources.push((
                    line[0].parse::<usize>().unwrap(),
                    line[1].parse::<NUM>().unwrap(),
                ));
            }
        };
        if x.chars().nth(0) == Some('a') {
            count += 1;
            let line = x.split(' ').collect::<Vec<&str>>()[1..].to_vec();
            let source: usize = line[0].parse::<usize>().unwrap();
            let target: usize = line[1].parse::<usize>().unwrap();
            let capacity: NUM = line[3].parse::<NUM>().unwrap();
            let cost: NUM = line[4].parse::<NUM>().unwrap();
            graph.add_edge(
                NodeIndex::new(source),
                NodeIndex::new(target),
                CustomEdgeIndices {
                    cost: (cost),
                    capacity: (capacity),
                    flow: (num_traits::zero::<NUM>()),
                },
            );
        };
    });
    (graph, sources, sinks)
}
