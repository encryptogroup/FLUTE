use std::fmt::Debug;
use std::path::{Path, PathBuf};
use tracing::error;
use walkdir::{DirEntry, WalkDir};

pub fn load_circuits<C, Err: Debug>(
    root: &Path,
    exclude: &[PathBuf],
    depth: Option<usize>,
    load_fn: impl Fn(&Path) -> Result<C, Err>,
) -> anyhow::Result<Vec<(PathBuf, C)>> {
    let mut circuits: Vec<_> = WalkDir::new(root)
        .max_depth(depth.unwrap_or(usize::MAX))
        .into_iter()
        .filter_entry(|entry| exclude.iter().all(|excl| excl != entry.path()))
        .filter_map(skip_dirs)
        .filter_map(|entry| {
            let res = load_fn(entry.path());
            match res {
                Ok(circ) => Some((entry.path().to_path_buf(), circ)),
                Err(err) => {
                    error!(?err, file = ?entry);
                    None
                }
            }
        })
        .collect();
    circuits.sort_by_cached_key(|(path, _)| path.clone());
    Ok(circuits)
}

fn skip_dirs(e: walkdir::Result<DirEntry>) -> Option<DirEntry> {
    match e {
        Err(err) => {
            error!("Error walking circuits directory. {:?}", err);
            None
        }
        Ok(entry) => {
            if entry.metadata().ok()?.is_dir() {
                None
            } else {
                Some(entry)
            }
        }
    }
}
