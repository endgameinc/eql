# EQL Contribution Guide
Welcome to the Event Query Language (EQL) contribution guide and thank you for expressing an interest in contributing to EQL!

As a quick refresher, the Event Query Language (EQL) was built by Endgame to express relationships between events. It is data source and platform agnostic, includes the ability to ask stateful questions, and enables sifting and stacking of data with unix-like pipes.

The EQL community consists of two main components
* This repository, which houses the underlying language and evaluation engine
* An [Analytics Library](https://eqllib.readthedocs.io/), which contains detections and hunting strategies.

Contributions to extend core capabilities of the language are directed to [``eql``](https://github.com/endgameinc/eql). For new detections, hunts, data sources, or knowledge sharing, please read the guidelines below before contributing.

# Table of Contents
1. [Contribution Process](#contribution-process)
2. [Ways to Contribute](#ways-to-contribute)
3. [Resources](#resources)
4. [Licenses](#licenses)

## Contribution Process
Contributing to EQL is a simple five-step process facilitated by Git:

1. Create an [issue](https://github.com/endgameinc/eqllib/issues) to track and discuss the work
2. Create a [branch](https://help.github.com/en/articles/about-branches)
3. Submit a [pull request](https://help.github.com/en/articles/about-pull-requests)
4. Update according to the code review
5. [Merge](https://help.github.com/en/articles/merging-a-pull-request) after approval.

### Additional Notes
* If you are accustomed to git, then great! If you aren't, don't fear, the command line tools are easy to use, but GitHub also has a straightforward process within your web browser to create branches and subsequent merging
* Use the Issues and PR templates! Git [Issues](https://github.com/endgameinc/eql/issues) are a great place to collaborate, discuss, or just track a request before development begins.
* There is plenty of literature and resources out there to help you. A great place to start is [GitHub guides](https://guides.github.com/).

## Ways to contribute
  
### Bug Fixes
Bug fixes are a natural area to contribute. We only ask that you please use the [bug report issue](https://github.com/endgameinc/eql/issues) to track the bug. Please elaborate on how to reproduce the bug and what behavior you would have expected. Compatibility is a priority for EQL, so be sure to capture information about your operating system and version of python. 

### Language or Engine Changes
For any changes within the language or the evaluation engine, propose your changes in a *Feature Request* issue to start a discussion. For new functionality function, be mindful of handling different edge cases, acceptable input, etc. We are happy to collaborate on such topics and encourage you to share ideas.

Some types of core development include:
* Syntax changes: This can be an entirely new construct ranging from high level constructs (e.g. sequence, join), to smaller changes (e.g. adding binary operators).
* Pipes: pipes usually can be added without needing to make syntax changes. Some pipes may stream output while processing, and others track state and only output when all input is finished (e.g. count).
* Functions: functions can be easily extended without needing to recommend changes to the syntax
* API: You may also want to add new APIs, utility functions, etc. like `is_stateful`

Anyone is encouraged to make a PR for open issues that have a clear path forward. When making changes, be sure to
* Link back to the issue "Resolves #100"
* Include unit tests in the relevant tests/ folder.
* Include end-to-end tests by updating the test [data](eql/etc/test_data.json) and [queries](eql/etc/test_queries.toml). These are used as the gold standard of expected behavior, and the queries should have a list of the serial_event_id of the events, in the expected order.

### CLI
Finally, the CLI is an area we are always looking to expand. This may include new input file types, new processing features, new tables, etc. Some shell functionality, like tab completions ANSI coloring, and history often varies across different operating systems. If possible, please test new functionality across a few different operating systems if you have access, and Python 2.7 and 3.6+. If you find any unusual behavior in the shell related to compatibility, please let us know in an issue.

## Resources
For additional resources on EQL, check [here](https://eql.readthedocs.io/en/latest/resources.html)

* Press Releases
  * [Public Availability of EQL](https://www.endgame.com/news/press-releases/endgame-announces-public-availability-eql)
* Blogs
  * [Introducing EQL](https://www.endgame.com/blog/technical-blog/introducing-event-query-language)
  * [EQL For the Masses](https://www.endgame.com/blog/technical-blog/eql-for-the-masses)
  * [Getting Started with EQL](https://www.endgame.com/blog/technical-blog/getting-started-eql)
* Conferences and Webinars
  * BlackHat 2019: Fantastic Red-Team Attacks and How to Find Them [(abstract)](https://www.blackhat.com/us-19/briefings/schedule/index.html#fantastic-red-team-attacks-and-how-to-find-them-16540)
  * BSIDES SATX 2019: The Hunter Games [(abstract)](https://www.bsidessatx.com/presentations-2019.html)
  * Circle City Con 2019: The Hunter Games [(video)](https://www.youtube.com/watch?v=K47gX3WHcm8)
  * Atomic Friday ([slides](https://eql.readthedocs.io/en/latest/_static/eql-crash-course.pdf)) ([video](https://www.youtube.com/watch?v=yvqxS5Bjc-s))
  * MITRE(TM) ATT&CK Con 2018: *From Technique to Detection*  [(video)](https://www.youtube.com/watch?v=a3hIIzJrH14)
* Read the Docs
  * [EQL](https://eql.readthedocs.io/)
  * [EQL Analytics Library](https://eqllib.readthedocs.io/)
* GitHub
  * [EQL](https://github.com/endgameinc/eql)
  * [EQL Analytics Library](https://github.com/endgameinc/eqllib)
  
## Licenses
The Event Query Language is licensed under [AGPL](LICENSE)
