# Collection of database scripts

This directory contains a collection of scripts that can be used to interact with the database:

* dump or copy databases
* generate a local copy of the model parameter database.

## Running a local copy of the model parameter database

The model parameter database is a mongoDB instance running on a server at DESY.
For testing and development, it might be useful to work with a local copy of the database.

**The following steps are "experimental" and need further testing.**

### Prerequisites

Access to a database dump of the production database is required.
