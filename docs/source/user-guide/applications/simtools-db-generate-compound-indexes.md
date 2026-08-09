# simtools-db-generate-compound-indexes

```{eval-rst}
.. automodule:: simtools.applications.db_generate_compound_indexes
   :members:
   :exclude-members: main
```

```{eval-rst}
This needs to be done once after a database has been set up.
Significantly accelerates database querying (at least a factor
of 5 in query time with a factor of 10 less documents examined).

**Command line arguments**
database_name (str, optional)
    Database name (use "all" for all databases)
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: db_generate_compound_indexes
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: db_generate_compound_indexes_run.yml
```
