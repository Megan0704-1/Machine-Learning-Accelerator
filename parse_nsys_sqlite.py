import sqlite3
import argparse
import sys

# Path to your SQLite database
DRAM_BW = 1935  # GB/s


def parseArgs():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0], description="Parse SQL db generator by the profiler."
    )
    parser.add_argument(
        "--sql_db",
        dest="sql_db_file",
        default=False,
        # type=text,
        help="Path to the sql lite file",
    )
    parser.add_argument(
        "--start_time",
        dest="st_t",
        default=False,
        type=float,
        help="Start time",
    )
    parser.add_argument(
        "--end_time",
        dest="end_t",
        default=False,
        type=float,
        help="End time",
    )

    args = parser.parse_args()

    return args


def get_DDR_BW(db_path, start_time, end_time):
    """
    Fetches the average read and write bandwidth from the sqlite report.

    Parameters:
        db_path (str): Path to the SQLite database file.
        start_time (float): start time mark
        end_time (float): end time mark
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # SQL query
        query_read = """
        SELECT timestamp, value 
        FROM GPU_METRICS 
        JOIN TARGET_INFO_GPU_METRICS USING (metricId) 
        WHERE timestamp > {} 
        AND timestamp < {}
        AND metricName == "DRAM Read Bandwidth [Throughput %]"
        """.format(
            start_time, end_time
        )

        # print(query_read)

        # Execute the SQL query
        cursor.execute(query_read)

        # Fetch all rows from the executed query
        results_read_bw = cursor.fetchall()

        values = [row[1] for row in results_read_bw]
        avg_rd_bw = sum(values) / len(values)

        # SQL query
        query_write = """
        SELECT timestamp, value 
        FROM GPU_METRICS 
        JOIN TARGET_INFO_GPU_METRICS USING (metricId) 
        WHERE timestamp > {} 
        AND timestamp < {}
        AND metricName == "DRAM Write Bandwidth [Throughput %]"
        """.format(
            start_time, end_time
        )

        # print(query_write)

        # Execute the SQL query
        cursor.execute(query_write)

        # Fetch all rows from the executed query
        results_write_bw = cursor.fetchall()

        values = [row[1] for row in results_write_bw]
        avg_wr_bw = sum(values) / len(values)

        # Return results
        return avg_rd_bw, avg_wr_bw

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
    finally:
        # Close the database connection
        if conn:
            conn.close()


def get_GA_active(db_path, start_time, end_time):
    """
    Fetches average GPU utilization from the report

    Parameters:
        db_path (str): Path to the SQLite database file.
        start_time (float): start time mark
        end_time (float): end time mark
    """

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # SQL query
        query = """
        SELECT timestamp, value 
        FROM GPU_METRICS 
        JOIN TARGET_INFO_GPU_METRICS USING (metricId) 
        WHERE timestamp > {} 
        AND timestamp < {}
        AND metricName == "SMs Active [Throughput %]"
        """.format(
            start_time, end_time
        )

        # print(query)

        # Execute the SQL query
        cursor.execute(query)

        # Fetch all rows from the executed query
        results = cursor.fetchall()

        values = [row[1] for row in results]
        avg = sum(values) / len(values)

        # Return results
        return avg

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
    finally:
        # Close the database connection
        if conn:
            conn.close()


def main():
    args = parseArgs()

    # Collect the arguments from argparse
    db_path = args.sql_db_file
    start_time = args.st_t * 1e9
    end_time = args.end_t * 1e9

    print(f"Select data from {start_time} timestamp to {end_time} timestamp")

    ga_active = get_GA_active(db_path, start_time, end_time)
    print("GPU utilization {:.2f} %".format(ga_active))

    rd_bw, wr_bw = get_DDR_BW(db_path, start_time, end_time)
    print(
        f"DRAM read bw {(rd_bw * DRAM_BW / 100):.2f} GB/s ({rd_bw:.2f} % of Total DRAM bw)"
    )
    print(
        f"DRAM write bw {(wr_bw * DRAM_BW / 100):.2f} GB/s ({wr_bw:.2f} % of Total DRAM bw)"
    )
    print(f"Total DRAM bw: {(rd_bw * DRAM_BW / 100 + wr_bw * DRAM_BW / 100):.2f} GB/s")


if __name__ == "__main__":
    main()
