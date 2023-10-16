using Pkg

using Documenter, StochasticAD, DocThemeIndigo

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
    ],
    "Public API" => "public_api.md",
    "Developer documentation" => "devdocs.md",
    "Limitations" => "limitations.md",
]

### Make docs

makedocs(sitename = "StochasticAD.jl",
    authors = "Gaurav Arya and other contributors",
    modules = [StochasticAD],
    format = format,
    pages = pages,
    warnonly = [:missing_docs])

try
    deploydocs(repo = "github.com/gaurav-arya/StochasticAD.jl",
        devbranch = "main",
        push_preview = true)
catch e
    println("Error encountered while deploying docs:")
    showerror(stdout, e)
end
