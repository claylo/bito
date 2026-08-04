use std::path::PathBuf;

use clap::Args;

#[derive(Args, Debug)]
pub struct ManArgs {
    /// Output directory (default: dist/share/man/man1)
    #[arg(long = "out-dir", default_value = "dist/share/man/man1")]
    pub out_dir: PathBuf,
}

pub fn cmd_man(args: ManArgs) -> Result<(), String> {
    let out_dir = crate::workspace_root().join(args.out_dir);

    // Renders the same augmented tree `librebar::cli::parse` uses, so the
    // pages cover `schema` and `completions` too. Filenames carry the full
    // hyphenated command path, so equally named leaves in different subtrees
    // cannot overwrite each other. Creates the output directory itself.
    let paths = librebar::cli::generate_manpages::<bito::Cli>(&out_dir)
        .map_err(|error| format!("generate manpages in {}: {error}", out_dir.display()))?;

    for path in paths {
        if is_help_page(&path) {
            std::fs::remove_file(&path).map_err(|error| format!("{}: {error}", path.display()))?;
            continue;
        }
        println!("wrote {}", path.display());
    }

    Ok(())
}

/// Whether this page documents clap's built-in `help` subcommand.
///
/// librebar's generator skips only *hidden* subcommands, and clap's
/// auto-generated `help` is not hidden — so it walks `help`, `help lint`,
/// `help custom show`, and so on, twenty pages of a subcommand explaining how
/// to read other subcommands. `.github/workflows/cd.yml` copies this whole
/// directory into the release archive, so they would ship.
///
/// Delete this filter once librebar skips built-in help subcommands itself.
fn is_help_page(path: &std::path::Path) -> bool {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| stem.split('-').any(|segment| segment == "help"))
}
