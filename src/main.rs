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

use clap::Parser;
use isera::*;
use std::marker::PhantomData;

use crate::dimacs_parser::parsed_graph;
mod dimacs_parser;

fn main() {
    //let args: Vec<String> = std::env::args().collect();
    //let mut rng = rand::thread_rng();

    #[derive(Parser, Debug)]
    #[command(version, about, long_about = None)]
    struct Args {
        /// path to file
        #[arg(short, long)]
        filename: String,
        /// nb of processors
        #[arg(short, long, default_value_t = 1)]
        nbproc: usize,
        /// kfactor for block size, only for block search pivot rule
        #[arg(short, long, default_value_t = 1)]
        kfactor: usize,
        /// pivot rule to use to find the entering arc
        #[arg(short, long, default_value = "block")]
        pivotrule: String,
    }

    let args = Args::parse();
    let file = args.filename.clone();
    let _file_path: String = args.filename;
    let nbproc = args.nbproc;
    let kfactor = args.kfactor;
    let mut pivotrulearg = args.pivotrule;
    if nbproc < 2 && pivotrulearg.contains("par") {
        pivotrulearg = "block".to_string();
    }
    let (graph, sources, sinks) = parsed_graph::<i64>(_file_path);

    print!("File: {:?}, ", file);
    let _min_cost_flow = match pivotrulearg.as_str() {
        "best" => min_cost(graph, sources, sinks, &pivotrules::BestEligible{phantom: PhantomData}, nbproc, kfactor),
        "parbest" => min_cost(graph, sources, sinks, &pivotrules::ParallelBestEligible{phantom: PhantomData}, nbproc, kfactor),
        "parblock" => min_cost(graph, sources, sinks, &pivotrules::ParallelBlockSearch{phantom: PhantomData}, nbproc, kfactor),
        "first" => min_cost(graph, sources, sinks, &pivotrules::FirstEligible{phantom: PhantomData}, nbproc, kfactor),
        _ => min_cost(graph, sources, sinks, &pivotrules::BlockSearch{phantom: PhantomData}, nbproc, kfactor),
    };

    print!("k = {:?}, nbproc = {:?}\n", kfactor, nbproc);
}
