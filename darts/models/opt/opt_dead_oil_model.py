import time
import math
import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime
from darts.engines import *


class OptDeadOilModel:
    def __init__(self):
        self.terminated_runs = 0
        self.objfun_norm = 1000
        self.opt_step_time = 0
        self.obj_time = 0
        self.sim_time = 0
        self.n_opt_steps = 0
        self.save_unfinished_runs = False

    def set_observation_data(self, data):
        # verify pandas format
        assert (type(data) == pd.core.frame.DataFrame)
        self.observation_data = data.set_index('time', drop=False)
        self.observation_last_date = data['time'][len(data['time']) - 1]

    def set_modifier(self, modifier):
        self.modifier = modifier

    def report(self):
        cum_data = pd.DataFrame.from_dict(self.physics.engine.time_data_report)
        return cum_data


    def set_objfun(self, objfun):
        self.objfun = objfun

    def gen_response_data (self):
        # generate response from time_data based on observed data
        assert (type(self.observation_data) == pd.core.frame.DataFrame)
        assert (type(self.response_data) == pd.core.frame.DataFrame)

        # convert time_data to df
        time_data = pd.DataFrame.from_dict(model.time_data)
        # generate time differences
        time_diff = time_data['time'].diff()
        time_diff[0] = time_data['time'][0]
        for c in self.observation_data:
            if c == 'time':
                continue
            volume = time_data[c] * time_diff


    def objfun_all_rates_l2(self):
        observation = self.observation_data
        response = self.response_data
        misfit = {}
        if len(response) == 0 or np.array(response['time'])[-1] != observation['time'][-1]:
            objfun = 1000
        else:
            for w in self.reservoir.inj_wells:
                c = w.name + ' : water rate (m3/day)'
                misfit[c] = observation[c] - response[c]

            for w in self.reservoir.prod_wells:
                c = w.name + ' : water rate (m3/day)'
                misfit[c] = observation[c] - response[c]
                c = w.name + ' : oil rate (m3/day)'
                misfit[c] = observation[c] - response[c]
        objfun = 0
        for m in misfit:
            if m != 'time':
                objfun += np.linalg.norm(misfit[m])
            else:
                continue
        return objfun / self.objfun_norm


    def objfun_observed_cumulative_rates_l2(self):
        # Get Data
        observation = self.observation_data
        time_data = self.physics.engine.time_data
        # seaching string
        search_str = 'rate (m3/day)'

        # Check for simullation to finish
        if len(time_data) == 0 or time_data['time'][len(time_data['time']) - 1] != self.observation_last_date:
            # simulation has not finished or even started
            return 1000
        else:

            # 1. Make simulated data cumulative
            time_data = pd.DataFrame.from_dict(time_data)
            time_data = self.make_cumulative(time_data)

            # 2. Get only relative data
            if time_data.size > observation.size:
                relevant_time_data = {key: time_data[key] for key in observation.columns}
            else:
                relevant_time_data = {key: time_data[key] for key in observation.columns if search_str in key or 'time' in key}
            response = pd.DataFrame.from_dict(relevant_time_data)

            # 2. Adjust response data accroding to uniform time
            response = response.set_index('time', drop=False)

            # 3. calculate misfit
            misfit = observation - response.loc[observation.index, :]

            # 4. Calculate objective function
            objfun = 0
            for m in misfit:
                if m != 'time' and search_str in m:
                    objfun += np.linalg.norm(misfit[m])

            return objfun / self.objfun_norm

    def objfun_observed_cumulative_rates_ver2(self):
        # Get Data
        observation = self.observation_data # already in cumulative format
        time_data = self.report()
        # seaching strings
        search_str_acc = 'acc volume'

        # Check for simullation to finish
        if len(time_data) == 0 or time_data['time'][len(time_data['time']) - 1] != self.observation_last_date:
            # simulation has not finished or even started
            print(time_data['time'][len(time_data['time']) - 1])
            print(self.observation_last_date)
            return 1000
        else:

            # 2. Get only relative data
            if time_data.size > observation.size:
                relevant_time_data = {key: time_data[key] for key in observation.columns}
            else:
                relevant_time_data = {key: time_data[key] for key in observation.columns if search_str_acc in key or 'time' in key}
            response = pd.DataFrame.from_dict(relevant_time_data)

            # 2. Adjust response data accroding to uniform time
            response = response.set_index('time', drop=False)

            # 3. calculate misfit
            if self.data_error_switch:
                data_error = self.calculate_trust_region(abs(observation),region=1) # region 2 is to check last period
                misfit = abs(abs(observation) - abs(response.loc[observation.index, :])) - data_error
                misfit[misfit < 0] = 0
            else:
                misfit = abs(abs(observation) - abs(response.loc[observation.index, :]))

            # 4. Calculate objective function
            objfun = 0
            for m in misfit:
                if m != 'time' and search_str_acc in m:
                        objfun += np.linalg.norm(misfit[m])

            return objfun / self.objfun_norm

    def objfun_observed_reactive_rates(self, start_opt=[],stop_opt=[]):
        # Get Data (if interval optimization choosen only spicific part of truth data will be extracted)
        if start_opt or stop_opt:
            observation = self.observation_data[start_opt:stop_opt]  # already in cumulative format # For Intervals
        else:
            observation = self.observation_data
        time_data = self.report()

        # seaching strings
        search_str = 'rate (m3/day)'

        # Check for simulation to finish
        if start_opt or stop_opt:
            if len(time_data) == 0: # For intervals last date is different TODO:can be optimized
                return 1000
            else:
                # 2. Get only relative data
                if time_data.size > observation.size:
                    relevant_time_data = {key: time_data[key] for key in observation.columns}
                else:
                    relevant_time_data = {key: time_data[key] for key in observation.columns if
                                          search_str in key or 'time' in key}
                response = pd.DataFrame.from_dict(relevant_time_data)

                # 2. Adjust response data accroding to uniform time
                # print(observation['time'])
                if len(response) !=len(observation['time']):
                    return 1000
                else:
                    response = response.set_index(observation['time'], drop=False)
        else:
            if len(time_data) == 0 or time_data['time'][len(time_data['time']) - 1] != self.observation_last_date:
                # simulation has not finished or even started
                print(time_data['time'][len(time_data['time']) - 1])
                print(self.observation_last_date)
                return 1000
            else:

                # 2. Get only relative data
                if time_data.size > observation.size:
                    relevant_time_data = {key: time_data[key] for key in observation.columns}
                else:
                    relevant_time_data = {key: time_data[key] for key in observation.columns if search_str in key or 'time' in key}
                response = pd.DataFrame.from_dict(relevant_time_data)

                # 2. Adjust response data accroding to uniform time
                response = response.set_index('time', drop=False) #

        # 3. calculate misfit
        misfit = abs(observation) - abs(response.loc[observation.index, :])
        misfit_std = sum(misfit.std())

        # 4. Calculate objective function - simple L2 norm of misfit
        objfun = 0
        for m in misfit:
            if m != 'time' and search_str in m and 'I' not in m:
                objfun += np.linalg.norm(misfit[m])

        # print(objfun)

        return objfun / self.objfun_norm


    def objfun_ensamble_based(self, start_opt=[],stop_opt=[]):
        # Get Data (if interval optimization choosen only spicific part of truth data will be extracted)
        if start_opt or stop_opt:
            observation = self.observation_data[start_opt:stop_opt]  # already in cumulative format # For Intervals
        else:
            observation = self.observation_data
        time_data = self.report()

        # Check for simulation to finish
        if start_opt or stop_opt:
            if len(time_data) == 0: # For intervals last date is different TODO:can be optimized
                return 1000
            else:
                # 2. Get only relative data
                if time_data.size > observation.size:
                    relevant_time_data = {key: time_data[key] for key in observation.columns}
                else:
                    relevant_time_data = {key: time_data[key] for key in observation.columns if
                                          search_str in key or 'time' in key}
                response = pd.DataFrame.from_dict(relevant_time_data)

                # 2. Adjust response data accroding to uniform time
                # print(observation['time'])
                if len(response) !=len(observation['time']):
                    return 1000
                else:
                    response = response.set_index(observation['time'], drop=False)
        else:
            if len(time_data) == 0 or time_data['time'][len(time_data['time']) - 1] != self.observation_last_date:
                # simulation has not finished or even started
                print(time_data['time'][len(time_data['time']) - 1])
                print(self.observation_last_date)
                return 1000
            else:

                # 2. Get only relative data
                if time_data.size > observation.size:
                    relevant_time_data = {key: time_data[key] for key in observation.columns}
                else:
                    relevant_time_data = {key: time_data[key] for key in observation.columns if search_str in key or 'time' in key}
                response = pd.DataFrame.from_dict(relevant_time_data)

                # 2. Adjust response data accroding to uniform time
                response = response.set_index('time', drop=False) #

        # 3. calculate misfit
        misfit = abs(observation) - abs(response.loc[observation.index, :])
        misfit_std = sum(misfit.std())

        # 4. Calculate objective function - based on covariance matrix
        # 4. Separate misfit into three components and combine either vertically or add up (make a single column vector)
        misfit_df = pd.DataFrame()
        misfit_df['time'] = misfit['time']
        misfit_df['Water Rate'] = 0
        misfit_df['Oil Rate'] = 0
        misfit_df['Injection Rate'] = 0
        search_str = ' : water rate'
        search_str_2 = ' : oil rate'

        for col in misfit.columns:
            if search_str in col:
                if 'I' not in col:
                    misfit_df['Water Rate'] += misfit[col]
                else:
                    misfit_df['Injection Rate'] += misfit[col]
            elif search_str_2 in col:
                if 'I' not in col:
                    misfit_df['Oil Rate'] += misfit[col]

        # misfit_df_single = pd.concat([misfit_df['Water Rate'], misfit_df['Oil Rate']], axis=0, ignore_index=True).values
        misfit_df_single = (misfit_df['Water Rate'] + misfit_df['Oil Rate']).values

        objfun = 0.5 * misfit_df_single.dot(self.cov_mat_inv).dot(np.transpose(misfit_df_single))


        return objfun / self.objfun_norm

    def objfun_watercut_l2(self):
        observed_response = self.get_response(self.observation_data)
        response = self.get_response(self.get_response_data())

        if len(response) == 0 or len(response[0]) == 0 or len(response[0][0]) != len(observed_response[0][0]):
            objfun = 1000
        else:
            objfun = 0
            all_water_rates = response[1]
            all_oil_rates = response[2]
            all_observed_water_rates = observed_response[1]
            all_observed_oil_rates = observed_response[2]
            for wrate, orate, wrate_obs, orate_obs in zip(all_water_rates, all_oil_rates, all_observed_water_rates,
                                                          all_observed_oil_rates):
                if True in np.isnan(wrate) or True in np.isnan(orate):
                    print('nan detected')
                    return 500
                else:
                    watercut = wrate / (wrate + orate)
                    watercut_obs = wrate_obs / (wrate_obs + orate_obs)
                    objfun += np.linalg.norm(watercut - watercut_obs) / np.linalg.norm(watercut_obs)

        return 10 * objfun

    # the function to be called by optimizer
    # 1. Update model according to model_modifier
    # 2. Perform run
    # 3. Return objective function
    def make_opt_step_stages(self, x, *args):
        self.opt_step_time -= time.time()

        # 1. Update model
        self.modifier.set_x(self, x)

        # 2. Reset
        self.reset(self.interval_init[args[2]])

        self.sim_time -= time.time()
        # 3. Run
        self.run(start_opt=args[0],stop_opt=args[1])
        # self.run()
        self.sim_time += time.time()

        self.obj_time -= time.time()
        # 4. Return objective
        obj = self.objfun(start_opt=args[0],stop_opt=args[1])
        self.obj_time += time.time()

        # 5. If simulation has not finished, rerun it to save the logs
        if self.save_unfinished_runs:
            if (obj == 1000):
                log_fname = 'terminated_run_%d' % self.terminated_runs

                with open(log_fname + '.x', 'w') as log:
                    log.write('Problem occurred with: \n')
                    log.write(np.array_str(x))

                np.save(log_fname, x)

                redirect_darts_output(log_fname + '.log')
                self.terminated_runs += 1
                self.modifier.set_x(self, x)
                self.reset()
                self.run()
                redirect_darts_output('')

        self.opt_step_time += time.time()
        self.n_opt_steps += 1

        print('\r Run %d: %f s/step, %f s/sim, %f s/obj' % (self.n_opt_steps, self.opt_step_time / self.n_opt_steps,
                                                              self.sim_time / self.n_opt_steps,
                                                              self.obj_time / self.n_opt_steps), end='',
              flush=True)
        return obj

    def make_opt_step(self, x, *args):
        # print(args[0])

        self.opt_step_time -= time.time()

        # 1. Update model
        self.modifier.set_x(self, x)

        # 2. Reset
        self.reset()

        self.sim_time -= time.time()
        # # 3. Run
        # self.run()
        # # self.run()
        # self.sim_time += time.time()
        #
        # self.obj_time -= time.time()
        # # 4. Return objective
        # obj = self.objfun()
        # self.obj_time += time.time()
        # 3. Run
        if args:
            args = args[0]
            self.run(start_opt=args[0],stop_opt=args[1])
        else:
            self.run()
        # self.run()
        self.sim_time += time.time()

        self.obj_time -= time.time()
        # 4. Return objective
        if args:
            obj = self.objfun(start_opt=args[0],stop_opt=args[1])
        else:
            obj = self.objfun()
            # print(obj)
        self.obj_time += time.time()

        # 5. If simulation has not finished, rerun it to save the logs
        if self.save_unfinished_runs:
            if (obj == 1000):
                log_fname = 'terminated_run_%d' % self.terminated_runs

                with open(log_fname + '.x', 'w') as log:
                    log.write('Problem occurred with: \n')
                    log.write(np.array_str(x))

                np.save(log_fname, x)

                redirect_darts_output(log_fname + '.log')
                self.terminated_runs += 1
                self.modifier.set_x(self, x)
                self.reset()
                self.run()
                redirect_darts_output('')

        self.opt_step_time += time.time()
        self.n_opt_steps += 1

        print('\r Run %d: %f s/step, %f s/sim, %f s/obj' % (self.n_opt_steps, self.opt_step_time / self.n_opt_steps,
                                                              self.sim_time / self.n_opt_steps,
                                                              self.obj_time / self.n_opt_steps), end='',
              flush=True)
        return obj

    def make_opt_step_2(self, x, grad, opt_control):
        args = opt_control
        eps = 1e-6 + 1e-10
        # eps = 1e-10
        if grad.size > 0:

            grad[:] = np.array(approx_fprime(x, self.make_opt_step, eps, opt_control))
            #scipy.optimize.check_grad
            # print(grad)

            self.opt_step_time -= time.time()

            # 1. Update model
            self.modifier.set_x(self, x)

            # 2. Reset
            self.reset()

            self.sim_time -= time.time()
            # # 3. Run
            # self.run()
            # self.sim_time += time.time()
            #
            # self.obj_time -= time.time()
            # # 4. Return objective
            # obj = self.objfun()
            # self.obj_time += time.time()
            # 3. Run
            self.run(start_opt=args[0], stop_opt=args[1])
            # self.run()
            self.sim_time += time.time()

            self.obj_time -= time.time()
            # 4. Return objective
            obj = self.objfun(start_opt=args[0], stop_opt=args[1])
            self.obj_time += time.time()

            # 5. If simulation has not finished, rerun it to save the logs
            if self.save_unfinished_runs:
                if (obj == 1000):
                    log_fname = 'terminated_run_%d' % self.terminated_runs

                    with open(log_fname + '.x', 'w') as log:
                        log.write('Problem occurred with: \n')
                        log.write(np.array_str(x))

                    np.save(log_fname, x)

                    redirect_darts_output(log_fname + '.log')
                    self.terminated_runs += 1
                    self.modifier.set_x(self, x)
                    self.reset()
                    self.run()
                    redirect_darts_output('')

            self.opt_step_time += time.time()
            self.n_opt_steps += 1

            print('\r Run %d: %f s/step, %f s/sim, %f s/obj' % (self.n_opt_steps, self.opt_step_time / self.n_opt_steps,
                                                                self.sim_time / self.n_opt_steps,
                                                                self.obj_time / self.n_opt_steps), end='',
                  flush=True)
        else:
            self.opt_step_time -= time.time()

            # 1. Update model
            self.modifier.set_x(self, x)

            # 2. Reset
            self.reset()

            self.sim_time -= time.time()
            # # 3. Run
            # self.run()
            # self.sim_time += time.time()
            #
            # self.obj_time -= time.time()
            # # 4. Return objective
            # obj = self.objfun()
            # self.obj_time += time.time()
            # 3. Run
            self.run(start_opt=args[0], stop_opt=args[1])
            # self.run()
            self.sim_time += time.time()

            self.obj_time -= time.time()
            # 4. Return objective
            obj = self.objfun(start_opt=args[0], stop_opt=args[1])
            self.obj_time += time.time()

            # 5. If simulation has not finished, rerun it to save the logs
            if self.save_unfinished_runs:
                if (obj == 1000):
                    log_fname = 'terminated_run_%d' % self.terminated_runs

                    with open(log_fname + '.x', 'w') as log:
                        log.write('Problem occurred with: \n')
                        log.write(np.array_str(x))

                    np.save(log_fname, x)

                    redirect_darts_output(log_fname + '.log')
                    self.terminated_runs += 1
                    self.modifier.set_x(self, x)
                    self.reset()
                    self.run()
                    redirect_darts_output('')

            self.opt_step_time += time.time()
            self.n_opt_steps += 1

            print('\r Run %d: %f s/step, %f s/sim, %f s/obj' % (self.n_opt_steps, self.opt_step_time / self.n_opt_steps,
                                                                self.sim_time / self.n_opt_steps,
                                                                self.obj_time / self.n_opt_steps), end='',
                  flush=True)
        print(f' Objective function: {obj}')
        return np.float(obj)

    def make_cumulative(self, darts_df):
        time_diff = darts_df['time'].diff()
        time_diff[0] = darts_df['time'][0]
        search_str = ' : water rate'
        search_str_2 = ' : oil rate'
        for col in darts_df.columns:
            if 'time' != col:
                darts_df[col] *= time_diff
                inj = darts_df[col] > 0
                prod = darts_df[col] < 0
                if sum(inj) > 0 and search_str in col:
                    darts_df[col + 'I'] = darts_df[col][inj]
                    darts_df[col + 'I'] = darts_df[col + 'I'].fillna(0)
                    darts_df[col + 'I'] = darts_df[col + 'I'].cumsum()
                if search_str in col or search_str_2 in col:
                    darts_df[col] = darts_df[col][prod]
                    darts_df[col] = darts_df[col].fillna(0)
                    darts_df[col] = darts_df[col].cumsum()

        return darts_df

    def make_cumulative_acc_format(self, darts_df):
        time_diff = darts_df['time'].diff()
        time_diff[0] = darts_df['time'][0]
        search_str = ' : water rate'
        search_str_2 = ' : oil rate'
        for col in darts_df.columns:
            if 'time' != col:
                for i, w in enumerate(self.reservoir.wells):
                    if search_str in col and w.name in col:
                        water_volume_col = w.name + " : water  volume (m3)"
                        water_acc_col = w.name + " : water  acc volume (m3)"
                        darts_df[water_volume_col] = darts_df[col] * time_diff
                        darts_df[water_acc_col] = darts_df[water_volume_col].cumsum()
                    elif search_str_2 in col and w.name in col:
                        oil_volume_col = w.name + " : oil  volume (m3)"
                        oil_acc_col = w.name + " : oil  acc volume (m3)"
                        darts_df[oil_volume_col] = darts_df[col] * time_diff
                        darts_df[oil_acc_col] = darts_df[oil_volume_col].cumsum()
        return darts_df


    def calculate_interval_error(self, start, stop, run_switch = False):
        # Get Data (if interval optimization choosen only spicific part of truth data will be extracted)
        if start or stop:
            observation = self.observation_data[start:stop]  # already in cumulative format # For Intervals
        else:
            observation = self.observation_data

        # seaching strings
        search_str = 'rate (m3/day)'

        if run_switch:
            self.run(export_to_vtk=False,start_opt=start,stop_opt=stop)
            print(1)

        time_data = self.report()[start:stop]

        # 2. Get only relative data
        if time_data.size > observation.size:
            relevant_time_data = {key: time_data[key] for key in observation.columns}
        else:
            relevant_time_data = {key: time_data[key] for key in observation.columns if
                                  search_str in key or 'time' in key}
        response = pd.DataFrame.from_dict(relevant_time_data)

        # 2. Adjust response data accroding to uniform time
        # print(observation['time'])
        if len(response) != len(observation['time']):
            return 1000
        else:
            response = response.set_index(observation['time'], drop=False)

        # 3. calculate misfit
        if self.data_error_switch:
            data_error = self.calculate_trust_region(abs(observation))  # region 2 is to check last period
            misfit = abs(abs(observation) - abs(response.loc[observation.index, :])) - data_error
            misfit[misfit < 0] = 0
        else:
            misfit = abs(abs(observation) - abs(response.loc[observation.index, :]))
        misfit['time']


            # 4. Calculate objective function
        objfun = 0
        objfun2 = 0
        for m in misfit:
            if m != 'time' and search_str in m:
                if self.data_error_switch:
                    objfun += np.linalg.norm(misfit[m])
                    objfun2 += np.linalg.norm(misfit[m], ord= np.inf)
                else:
                    # objfun += np.linalg.norm(misfit[m] / math.sqrt(np.std(misfit[m])))
                    # objfun += sum(misfit[m]**2)
                    # objfun += sum(misfit[m])
                    # objfun += np.linalg.norm(misfit[m])
                    # objfun += np.linalg.norm(misfit[m] / math.sqrt(np.var(misfit[m])))
                    # objfun += 1/len(misfit[m]) * sum((misfit[m]**2)/ np.std(misfit[m]))
                    objfun += 1/len(misfit[m]) * sum((misfit[m]**2))


        return objfun / self.objfun_norm, objfun2 / self.objfun_norm

    def calculate_misfit_error(self, truth_df, opt_df, train_lenght, rate_type):

        search_str = rate_type

        if opt_df.size > truth_df.size:
            relevant_time_data = {key: opt_df[key] for key in truth_df.columns}
        else:
            relevant_time_data = {key: opt_df[key] for key in truth_df.columns if
                                  search_str in key or 'time' in key}
        response = pd.DataFrame.from_dict(relevant_time_data)

        search_str = ' : water rate'
        search_str_2 = ' : oil rate'

        acc_df = pd.DataFrame()
        acc_df['time'] = truth_df['time']
        acc_df['Water Rate'] = 0
        acc_df['Oil Rate'] = 0
        acc_df['Injection Rate'] = 0
        acc_df_2 = pd.DataFrame()
        acc_df_2['time'] = truth_df['time']
        acc_df_2['Water Rate'] = 0
        acc_df_2['Oil Rate'] = 0
        acc_df_2['Injection Rate'] = 0
        for col in truth_df.columns:
            if search_str in col:
                if 'I' not in col:
                    acc_df['Water Rate'] += truth_df[col]
                else:
                    acc_df['Injection Rate'] +=truth_df[col]
            elif search_str_2 in col:
                if 'I' not in col:
                    acc_df['Oil Rate'] += truth_df[col]
        for col in opt_df.columns:
            if search_str in col:
                if 'I' not in col:
                    acc_df_2['Water Rate'] += opt_df[col]
                else:
                    acc_df_2['Injection Rate'] +=opt_df[col]
            elif search_str_2 in col:
                if 'I' not in col:
                    acc_df_2['Oil Rate'] += opt_df[col]

        # acc_df['Water Rate'][acc_df['Water Rate'] == 0] = 1e-6
        # SAE = abs(abs(acc_df['Water Rate'])-abs(acc_df_2['Water Rate']))
        SAE = abs(abs(acc_df['Oil Rate'])-abs(acc_df_2['Oil Rate']))

        MAE_1 = (SAE[0:train_lenght].sum())/len(SAE[0:train_lenght])
        MAE_2 = (SAE[train_lenght:].sum())/len(SAE[train_lenght:])


        # AE = abs((abs(acc_df_2['Water Rate'])-abs(acc_df['Water Rate']))/abs(acc_df['Water Rate']))*100
        # AE = AE.replace(np.inf, 0)
        AE = abs((abs(acc_df_2['Oil Rate'])-abs(acc_df['Oil Rate']))/abs(acc_df['Oil Rate']))*100


        # misfit = abs(abs(truth_df) - abs(response.loc[truth_df.index, :]))
        #
        # error_training = 0
        # error_prediction = 0
        # for m in misfit:
        #     if m != 'time' and search_str in m and 'I' not in m:
        #         # error_training += misfit[m][0:train_lenght+1].sum()
        #         # error_prediction += misfit[m][train_lenght+1:].sum()
        #         error_training += np.linalg.norm(misfit[m][0:train_lenght+1])
        #         error_prediction += np.linalg.norm(misfit[m][train_lenght+1:])

        return MAE_1,MAE_2, np.mean(AE[0:train_lenght]), np.mean(AE[train_lenght:])






