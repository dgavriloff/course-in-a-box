---
title: Understanding the Course Template
---

## Understanding the Course Template
<br>

<center><img src="/img/course-template.png" alt="Screenshot of the course template in GitHub" width="75%"/></center>

<br>

Once you’ve forked the course, you’ll see the course's guts as a list of folders and file names. This list contains all of the learning content for the course as well as the code that holds it all together. This guide will always tell you exactly where to look to update different parts of your course, so you don’t need to read or memorize this section in order to proceed.

<br>

#### Directory layout explanation

```
📁 _data - course metadata 
📁 _includes - html for page components
📁 _layouts - html for various page layouts
📁 _sass - styling code
📁 css - more styling code
📁 fonts - font files (including icons!)
📁 img - image files
📁 js - JavaScript for interactive components of the site
📁 modules - folder for the actual course content
.gitignore - a git file (for keeping this directory tidy)
404.html - error text that appears on broken links or nonexistent pages
CNAME - use for setting custom domains
docker-compose.yml - configuration for running Jekyll locally with Docker
Gemfile - used by jekyll/GitHub pages
LICENSE - Course-in-a-Box is licensed under the MIT License (do not edit)
README.md - a text file to introduce and explain the project
_config.yml - Jekyll/GitHub pages configuration & settings
index.md - landing page content
single-page.md - an example single page of content
```
