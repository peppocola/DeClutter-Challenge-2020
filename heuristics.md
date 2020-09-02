# Heuristics

For calculating the Jaccard index and the position, we need to get the line a comment refers to.
Our assumption is that a comment usually refers to the following line. The sense of comments is to get some basic understanding what the code does, without the need to read the code itself. So it makes much more sense to place the comments before the code it describes. [1]

Here are the heuristics used to find the line.
1. If the line in which the comment is written is not empty (so the line contains more information than just the comment), we select that line.

2. If the line is empty or contains something like else, try or finally, we go on analyzing the following line while they're empty.
A line containing just an else, try or finally keyword doesn't contain useful information. So we decide to keep searching around.

3. If the comment is written inside an empty block, we take as referring line the first non-empty line preceding the comment. If the block is empty and there's just a comment inside, probabily the comment refers to an eventual condition preceding the block.

4. If analyzing the lines that follow the comment we find the end of the block ("}"), we start searching a not-empty line preceding the comment, following the previous heuristics but going backwards in the code.

5. If we get to EOF or BOF we select an empty string.


[1] https://softwareengineering.stackexchange.com/questions/126601/comment-before-or-after-the-relevant-code
