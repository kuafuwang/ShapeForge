# Metal sources

These two files are Apple's documents, downloaded unchanged on 2026-10-20 from [Metal Resources](https://developer.apple.com/metal/resources/). They are not a rewrite and not a blog summary.

| File | What it is | This copy |
| --- | --- | --- |
| `Metal-Shading-Language-Specification.pdf` | The language: syntax, address spaces, attributes, the standard library, `simdgroup`, barriers. Apple's resources page calls it the definitive MSL specification. The same URL's text begins "Metal Shading Language Specification Version 4.1". | 383 pages, 14 MB |
| `Metal-Feature-Set-Tables.pdf` | Which GPU family has which feature and which limits. M4 is Apple9. | 18 pages, 2.9 MB |

Order when two sentences disagree:

1. A number this machine measured (`tcenv calibrate`, an evaluation report, or a fenced example in `msl_reference.md` that this compiler accepted).
2. These two PDFs.
3. An Apple WWDC session.
4. A blog. A blog is a lead. It does not go in as a fact.

`msl_reference.md` records this compiler as MSL 4.0 on an M4 Max (Apple9). The specification file may describe 4.1. A 4.1-only feature is not available until this compiler accepts it.
