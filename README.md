# An open-source project for creating online courses, built by P2PU
Course-in-a-Box is a free tool for building and publishing online courses — no prior coding experience required. 

To create your own course, simply fork this repository and delete the CNAME file. Detailed documentation is available at [course-in-a-box.p2pu.org](https://course-in-a-box.p2pu.org).

To make changes to the template itself, a good place to start is the [`_layouts`](/_layouts), [`_includes`](/_includes) and [`css`](/css) directories. These directories contain all the layout and style files used.

Questions? Ask on P2PU's [Community Forum](https://community.p2pu.org/c/tech/course-in-a-box/78).

# Running locally
- [Install Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/)
- Run `docker compose up` (or `USER_ID=$(id -u) USER_GID=$(id -g) docker compose up` on Linux if you encounter permission errors)
- Your site will be available at [http://localhost:4000](http://localhost:4000)

Alternatively, you can run without docker-compose:
```bash
docker run -i -t --rm -u $(id -u):$(id -g) -p 4000:4000 -v `pwd`:/opt/app -v `pwd`/.bundler/:/opt/bundler -e BUNDLE_PATH=/opt/bundler -w /opt/app ruby:3.3 bash -c "bundle install && bundle exec jekyll serve --watch -H 0.0.0.0"
```

---
Course-in-a-Box is built by [Peer 2 Peer University](https://www.p2pu.org) and shared under an MIT License.

Course content ("Modules") are shared under a [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).
