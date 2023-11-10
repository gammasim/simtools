.. _pull_requests:

Pull requests
*************

Pull requests are one of our most important and powerful development tools.
They help to obtain good quality, homogenous, and maintainable code.

Both reviewing pull requests and addressing reviewers' comments is hard.
The following thoughts might help to obtain an efficient and pleasant reviewing process.

Before issuing a pull request
-----------------------------

- changes and new features shall be discussed with the team before implementation
- all tests shall pass
- no conflicts with 'main' (best to do a `merge main` before the pull request)
- self-review the changes made (e.g., by using the draft pull request feature)

Issuing a pull request
----------------------

- describe clearly the problem or issue solved by the pull request; do not assume full familiarity of the reviewer with the issue
- indicate if this a new feature, refactoring, or bug fix
- describe the main logic of the changes
- guide the reviewer through the changes; indicate for larger pull requests a logical order for reviewing
- consider adding code or usage examples
- add any statement helping reviewers (e.g., indicate which files to review first)
- pull request should focus on one problem and should be short (even though it is easier to make big ones)
- if the problem to be solved requires many changes: break down the problem in logical pieces; explain the breakdown of the problem in an issue


Reviewing a pull request
------------------------

- seek to understand the purpose of the pull request; reach out to the author before reviewing if it is not clear which problem will be solved
- if you reject the pull request or if you think another approach might be more efficient, request immediately (not after many other patches are implemented); if there is a major issue, focus on that first and indicate that detailed comments will come at a later point
- indicate on the pull request page (or e.g., in the chat) that you have started reviewing to avoid duplicated reviews
- propose solutions, not just disagreement; reviewer comments should lead to improvements of the pull request (e.g., a comment saying "I don't like this" is not helping)
- accept that there are different solutions (keeping the homogeneity of the code in mind)
- focus on relevant points
- discuss the code, not the coder (it is "the code is doing..", not "you are doing.."). Often logic has been introduced earlier and there is no use of pointing fingers.
- pull the code locally and execute the new code
- limit the review to the changes in the pull request. Open an issue or a follow-up pull request for anything else.
- clearly indicate if you have no time for a review (this is ok if it is known)
- fix small typos directly
