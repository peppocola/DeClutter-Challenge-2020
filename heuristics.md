# Heuristics for identifying the code block a comment refers to 

In the following, we describe the heuristics we implemented to identify the code block a comment refers to. This information is used to calculate the Jaccard index and the position, used by the classifier described in 

> Giuseppe Colavito, Pierpaolo Basile, Nicole Novielli (2020). "Leveraging Textual and Non-Textual Features forDocumentation Decluttering". In _Proceedings of the Second Software Documentation Generation Challenge (DocGen2)_, co-located with ICSME 2020.

Our assumption is that a comment usually refers to the following code line/block. In fact, the purpose of comments is to provide an explanation of what the code does, without the need to read the code itself [1]

We define and implement the following heuristics used to find the code line (either the individual statement or the starting line of a code block).
1. If the line in which the comment is written is not empty, i.e. the line contains the comment and the statement it refers to, we select that line.

2. If the line contains the comment only or also includes keywords as _else_, _try_ or _finally_, we assume the comment refers to the code block following the comment line.

3. If the comment is written inside an empty block or is immediately foolwed the end of the block  ("}"), we assume the comment refers to the immediately preceding code block. 

[1] https://softwareengineering.stackexchange.com/questions/126601/comment-before-or-after-the-relevant-code
