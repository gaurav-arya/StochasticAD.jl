using Pkg

using Documenter
using StochasticAD
using DocThemeIndigo
using Literate

### Formatting

indigo = DocThemeIndigo.install(StochasticAD)
format = Documenter.HTML(prettyurls = false,
    assets = [indigo, "assets/extra_styles.css"],
    repolink = "https://github.com/gaurav-arya/StochasticAD.jl",
    edit_link = "main")

### Pagination

pages = [
    "Overview" => "index.md",
    "Tutorials" => [
        "tutorials/random_walk.md",
        "tutorials/game_of_life.md",
        "tutorials/particle_filter.md",
        "tutorials/optimizations.md",
        "tutorials/reverse_demo.md"
    ],
    "Public API" => "public_api.md",
    "Developer documentation" => "devdocs.md",
    "Limitations" => "limitations.md"
]

### Prepare literate tutorials

# TODO (for now they are manually built into docs/src/tutorials and checked into repo)

### Make docs

makedocs(sitename = "StochasticAD.jl",
    authors = "Gaurav Arya and other contributors",
    modules = [StochasticAD],
    format = format,
    pages = pages,
    warnonly = [:missing_docs, :cross_references],
    draft = true)

try
    deploydocs(repo = "github.com/gaurav-arya/StochasticAD.jl",
        devbranch = "main",
        push_preview = true)
catch e
    println("Error encountered while deploying docs:")
    showerror(stdout, e)
end
