# Project rules

## Comments

Only write a comment that states something the code cannot. If a reader can infer
it by reading the line, the comment must not exist.

Do not write:
- comments that restate the code (`# loop over rows` above a `for` loop)
- comments that label a step (`# build the model`, `# now train`)
- comments explaining a well-known API or idiom
- docstrings that narrate the implementation line by line
- rationale for a decision that is already obvious from the surrounding code

Do write, and keep it to one line where possible:
- a non-obvious constraint or invariant a reader would otherwise violate
- why a surprising or wrong-looking line is actually correct
- a unit, bound, or convention not visible in the identifier

When in doubt, leave it out. Fewer comments is the default, not a tradeoff.
