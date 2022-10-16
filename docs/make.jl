using Pkg
Pkg.add(url = "https://github.com/JuliaDocs/Documenter.jl") # Documenter.jl has some unreleased features I want. TODO: remove

using Documenter, StochasticAD, DocThemeIndigo

### Formatting

indigo = DocThemeIndigo.install(StochasticAD)
format = Documenter.HTML(prettyurls = false,
                         assets = [indigo],
                         repolink = "https://github.com/gaurav-arya/StochasticAD.jl",
                         edit_link = "main")

### Pagination

pages = [
    "Introduction" => "index.md",
    "Tutorials" => [
        "tutorials/random_walk.md",
        "tutorials/particle_filter.md",
    ],
]

### Make docs

makedocs(sitename = "StochasticAD.jl",
         authors = "Gaurav Arya and other contributors",
         modules = [StochasticAD],
         format = format,
         pages = pages,
         strict = [
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
         ])

deploydocs(repo = "github.com/gaurav-arya/StochasticAD.jl",
           devbranch = "main",
           push_preview = true)
